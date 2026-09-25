"""Build frozen Stage-1 corrector caches and train-only residual statistics."""
from __future__ import annotations
import argparse, json, hashlib, time
from pathlib import Path
import numpy as np, torch

from ..data.cache import CachedWeatherDataset
from .corrector_data import write_phase_a_cache, CorrectorCacheDataset, ResidualStatsAccumulator
from ..models.physics import unit_sphere_cell_areas
from ..models.predictor_adapter import PredictorAdapter
from ..models.predictor import StructuredPredictor

ROOT=Path(__file__).parents[2]

def load_predictor(device, checkpoint):
    audit=json.loads((ROOT/'results/data_audit.json').read_text());stats=audit['normalization'];lat=torch.tensor(audit['grid']['latitude'])
    model=StructuredPredictor(lat.deg2rad(),stats['state_std'],stats['delta_std']).to(device)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state['model'],strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return PredictorAdapter(model),audit

def build(config_path, batch_size=128):
    cfg=json.loads(Path(config_path).read_text()); checkpoint=ROOT/cfg['predictor_checkpoint']
    device=torch.device('cuda');predictor,audit=load_predictor(device,checkpoint);outroot=ROOT/cfg['data']['cache_root'];outroot.mkdir(parents=True,exist_ok=True)
    def infer(batch):
        b={k:v.to(device,non_blocking=True) for k,v in batch.items() if k not in {'target','calendar','year'}}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return predictor(b['model_input'].float(),b['current'].float(),b['current_climatology'].float(),b['target_climatology'].float())
    outputs=[]
    for split in ('train','val'):
        dest=outroot/split
        if not dest.exists(): outputs.append(str(write_phase_a_cache(CachedWeatherDataset(ROOT/'cache/era5'/split),outroot,split,infer,batch_size=batch_size)))
        else: outputs.append(str(dest))
    train=CorrectorCacheDataset(outroot/'train');lat=torch.tensor(audit['grid']['latitude']).deg2rad();area=unit_sphere_cell_areas(lat.double(),64).float();acc=ResidualStatsAccumulator(area)
    for left in range(0,len(train),256):
        right=min(len(train),left+256);target=torch.from_numpy(np.asarray(train.arrays['target'][left:right]));base=torch.from_numpy(np.asarray(train.arrays['base_prediction'][left:right]));acc.update(target-base)
    stats_path=ROOT/cfg['data'].get('residual_stats','results/predictor_residual_stats.npz');sha=acc.save(stats_path)
    meta={'status':'complete','config':str(config_path),'cache_dirs':outputs,'samples':{'train':len(train),'val':len(CorrectorCacheDataset(outroot/'val'))},'residual_stats':str(stats_path),'sha256':sha,'predictor_checkpoint':str(checkpoint),'predictor_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    manifest=ROOT/cfg['data'].get('cache_manifest','results/corrector_cache_manifest.json');manifest.write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--batch-size',type=int,default=128);a=p.parse_args();build(a.config,a.batch_size)
