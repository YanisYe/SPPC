"""One-shot test evaluation of a validation-selected corrector EMA."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from ..data.cache import CachedWeatherDataset
from .corrector_protocol import evaluate_corrector,FINAL_LEADS
from ..training.corrector_objectives import CorrectorEMA
from ..training.train_corrector import ROOT,load_payloads,load_predictor,build_corrector

def evaluate(config_path,checkpoint,output,batch_size=32):
 out=Path(output)
 if out.exists():raise FileExistsError(out)
 cfg,audit,stats,lat,res=load_payloads(config_path);dev=torch.device('cuda');predictor,_=load_predictor(cfg,stats,lat,dev);corrector=build_corrector(cfg,stats,lat,res,dev)
 state=torch.load(checkpoint,map_location='cpu',weights_only=False)
 if 'joint_ema' not in state:raise ValueError('final corrector checkpoint lacks joint_ema')
 ema=CorrectorEMA({'predictor':predictor,'corrector':corrector});ema.load_state_dict(state['joint_ema'])
 test=CachedWeatherDataset(ROOT/'cache/era5/test')
 with ema.apply({'predictor':predictor,'corrector':corrector}):
  result=evaluate_corrector(predictor,corrector,test,stats,lat,split='test',leads=FINAL_LEADS,batch_size=batch_size,device=dev)
 result.update(status='test',variant='structured_corrector',checkpoint=str(checkpoint),selection='validation_only_test_once',years=[2017,2018])
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,allow_nan=False));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args();evaluate(a.config,a.checkpoint,a.output,a.batch_size)
