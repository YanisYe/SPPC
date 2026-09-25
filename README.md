# SPPC

SPPC is a **Structure-Preserving Predictor-Corrector** for data-driven weather forecasting on a spherical grid. It combines a neural predictor with explicit physical operators and a structured residual corrector for autoregressive forecasts.

The provided configuration uses 6-hour ERA5 data on a `32 x 64` latitude-longitude grid and predicts five variables:

- 2 m temperature (`T2m`)
- 850 hPa temperature (`T850`)
- 500 hPa geopotential (`Z500`)
- 10 m zonal wind (`U10`)
- 10 m meridional wind (`V10`)

## Method

The predictor combines a spherical neural backbone with Hodge wind projection, geostrophic wind estimation, semi-Lagrangian transport, and conservative shared-edge divergence. The corrector learns structured residual updates for scalar fields and wind.

Training has three stages:

1. Train the single-step structured predictor.
2. Train the corrector on frozen predictor outputs.
3. Jointly fine-tune the corrector and predictor tail over four-step rollouts.

## Repository layout

```text
configs/          Training configurations
checkpoints/      Final pretrained SPPC checkpoint (Git LFS)
sppc/data/        ERA5 validation, statistics, and cache preparation
sppc/models/      Predictor, corrector, spherical backbone, and physics operators
sppc/training/    Training pipelines and objectives
sppc/evaluation/  Rollout evaluation and metrics
sppc/evaluate.py  Evaluation entry point for the bundled checkpoint
```

## Installation

Python 3.11 or newer and an NVIDIA GPU are required. The reference environment uses PyTorch 2.6 with CUDA 12.4.

```bash
git clone https://github.com/YanisYe/SPPC.git
cd SPPC
git lfs pull

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e .
```

Each command below expects exactly one visible CUDA GPU.

## Data preparation

Arrange ERA5 data as follows:

```text
ERA5_ROOT/
|-- lat.npy
|-- lon.npy
|-- train/   # 2006-2015
|-- val/     # 2016
`-- test/    # 2017-2018
```

Each split must contain chronological `YEAR_SHARD.npz` files stored without ZIP compression. Every variable must have shape `[time, 1, 32, 64]` and use these array names:

```text
2m_temperature
temperature_850
geopotential_500
10m_u_component_of_wind
10m_v_component_of_wind
u_component_of_wind_850
v_component_of_wind_850
u_component_of_wind_500
v_component_of_wind_500
```

Audit the source data and build memory-mapped caches:

```bash
mkdir -p results cache checkpoints

python -m sppc.data.audit \
  --source-root /absolute/path/to/ERA5_ROOT \
  --output results/data_audit.json

python -m sppc.data.cache \
  --source-root /absolute/path/to/ERA5_ROOT \
  --cache-root cache/era5 \
  --stats-json results/data_audit.json
```

## Training

Train the predictor:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sppc.training.train_predictor \
  --config configs/ssp500.json \
  --no-resume
```

Promote the best validation checkpoint:

```bash
python - <<'PY'
import json
import shutil
from pathlib import Path

root = Path("checkpoints/ssp500")
rows = json.loads((root / "top3/index.json").read_text())
best = min(rows, key=lambda row: (float(row["score"]), int(row["epoch"])))
shutil.copy2(root / "top3" / best["checkpoint"], root / "best.pt")
print(root / "best.pt")
PY
```

Prepare predictor outputs and train the corrector through Phases A and B:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sppc.training.prepare_corrector \
  --config configs/structured_corrector_a300_b200.json \
  --batch-size 128

CUDA_VISIBLE_DEVICES=0 python -m sppc.training.train_corrector \
  --config configs/structured_corrector_a300_b200.json \
  --phase all
```

## Evaluation

The bundled checkpoint contains the final predictor-corrector EMA weights and normalization metadata. After preparing `cache/era5/test`, evaluate 6-36 hour lead times with:

```bash
CUDA_VISIBLE_DEVICES=0 python -m sppc.evaluate
```

The default output is `results/sppc_test_6_36.json`. The output path and batch size can be changed with `--output` and `--batch-size`.

## License

This project is released under the [MIT License](LICENSE).
