"""Execute the final Structured Corrector Phase A and Phase B pipeline."""
from __future__ import annotations
import argparse, hashlib, json, math, os, time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.utils.data import DataLoader
from ..data.cache import CachedWeatherDataset
from .common import atomic_json_write, atomic_torch_save, learning_rate_for_update, optimizer_parameter_groups, set_reproducibility
from .checkpoints import configured_epochs, phase_a_checkpoint_path, phase_status_payload
from ..models.corrector import StructuredCorrector
from .corrector_data import CorrectorCacheDataset, FourStepCorrectorDataset
from ..evaluation.rollout import evaluate_corrector, choose_phase_a_checkpoint, choose_phase_b_checkpoint, FINAL_LEADS
from .corrector_objectives import CorrectorEMA, PhaseAProtocol, PhaseBProtocol, accumulation_plan, gradients_are_finite, phase_a_objective, phase_b_rollout_objective, set_phase_a_trainable, set_phase_b_trainable
from ..evaluation.metrics import evaluate_forecast
from ..models.predictor_adapter import PredictorAdapter
from ..models.predictor import StructuredPredictor

ROOT=Path(__file__).parents[2]

def load_payloads(config_path):
 cfg=json.loads(Path(config_path).read_text());audit=json.loads((ROOT/'results/data_audit.json').read_text());stats=audit['normalization'];lat=torch.tensor(audit['grid']['latitude'],dtype=torch.float32);res=np.load(ROOT/cfg['data'].get('residual_stats','results/predictor_residual_stats.npz'));return cfg,audit,stats,lat,res

def build_predictor(stats,lat,device):
 return PredictorAdapter(StructuredPredictor(lat.deg2rad(),stats['state_std'],stats['delta_std']).to(device))

def load_predictor(cfg,stats,lat,device):
 m=build_predictor(stats,lat,device);p=ROOT/cfg['predictor_checkpoint'];s=torch.load(p,map_location='cpu',weights_only=False);m.predictor.load_state_dict(s['model'],strict=True);return m,p

def build_corrector(cfg,stats,lat,res,device):
 kw=dict(state_mean=stats['state_mean'],state_std=stats['state_std'],std_dx=stats['delta_std'],std_res_wind=res['std_res_wind'],latitudes=lat.deg2rad())
 if cfg.get('corrector') != 'full': raise ValueError('SPPC only supports StructuredCorrector')
 return StructuredCorrector(std_res_zero=res['std_res_zero'],std_res_mean=res['std_res_mean'],**kw).to(device)

def collate(batch):
 return {k:torch.from_numpy(np.stack([np.asarray(x[k]) for x in batch])) if isinstance(batch[0][k],np.ndarray) else torch.stack([x[k] for x in batch]) for k in batch[0]}

def optimizer_for_corrector(c, cfg):
 phase=cfg['phase_a'];return torch.optim.AdamW(optimizer_parameter_groups(c,float(phase['weight_decay_matrix'])),lr=float(phase['learning_rate']),betas=tuple(phase['betas']),eps=float(phase['eps']),fused=True)

def ema_weights(ema,prefix):
 return {k[len(prefix)+1:]:v for k,v in ema.shadow.items() if k.startswith(prefix+'.')}

def annual_climate(dataset):
 years=np.asarray(dataset.arrays['year']);out={}
 for y in np.unique(years):
  ids=np.flatnonzero(years==y);seq=np.concatenate((np.asarray(dataset.arrays['current'][ids]),np.asarray(dataset.arrays['target'][ids[-1]])[None]));out[int(y)]=seq.mean(0,dtype=np.float64).astype(np.float32)
 return out

@torch.inference_mode()
def eval_phase_a(c,dataset,stats,lat,device,batch=32):
 c.eval();pred=[];truth=[];clim=[];annual=annual_climate(dataset);years=np.asarray(dataset.arrays['year'])
 for left in range(0,len(dataset),batch):
  right=min(len(dataset),left+batch);cur=torch.from_numpy(np.asarray(dataset.arrays['current'][left:right])).to(device);base=torch.from_numpy(np.asarray(dataset.arrays['base_prediction'][left:right])).to(device);feat=torch.from_numpy(np.asarray(dataset.arrays['decoder_features'][left:right])).to(device)
  with torch.autocast('cuda',dtype=torch.bfloat16): out=c(cur,base,feat)
  pred.append((base+out['correction']).float().cpu());truth.append(torch.from_numpy(np.asarray(dataset.arrays['target'][left:right])).clone());clim.append(torch.from_numpy(np.stack([annual[int(y)] for y in years[left:right]])))
 return evaluate_forecast(torch.cat(pred),torch.cat(truth),torch.cat(clim),lat,torch.tensor(stats['state_std']))

def base_validation(cfg,stats,lat,device,leads=FINAL_LEADS):
 predictor,_=load_predictor(cfg,stats,lat,device);ds=CachedWeatherDataset(ROOT/'cache/era5/val')
 return evaluate_corrector(predictor,ZeroCorrector().to(device),ds,stats,lat,split='val',leads=leads,batch_size=32,device=device)['metrics']

class ZeroCorrector(torch.nn.Module):
 def forward(self,current,base,decoder_features):return {'correction':torch.zeros_like(base)}

def phase_a(config_path):
 cfg,audit,stats,lat,res=load_payloads(config_path);device=torch.device('cuda');set_reproducibility(int(cfg['seed']));predictor,predictor_path=load_predictor(cfg,stats,lat,device);c=build_corrector(cfg,stats,lat,res,device);set_phase_a_trainable(predictor,c)
 train=CorrectorCacheDataset(ROOT/cfg['data']['cache_root']/'train');val=CorrectorCacheDataset(ROOT/cfg['data']['cache_root']/'val');micro=int(cfg['phase_a']['micro_batch_size']);accum,effective=accumulation_plan(int(cfg['phase_a']['effective_global_batch_size']),micro);opt=optimizer_for_corrector(c,cfg);ema=CorrectorEMA({'corrector':c},float(cfg['phase_a']['ema_decay']));epochs=configured_epochs(cfg,'phase_a');steps=math.ceil(math.ceil(len(train)/micro)/accum);total=epochs*steps;warm=int(cfg['phase_a']['warmup_epochs'])*steps;run=ROOT/cfg['outputs']['run_dir'];run.mkdir(parents=True,exist_ok=True);ledger=run/'phase_a_status.json';records=[];global_step=0
 for epoch in range(1,epochs+1):
  c.train();loader=DataLoader(train,batch_size=micro,shuffle=True,generator=torch.Generator().manual_seed(epoch),num_workers=0,drop_last=False,collate_fn=collate);opt.zero_grad(set_to_none=True);loss_sum=0.;seen=0;t0=time.perf_counter()
  for bi,b in enumerate(loader):
   cur=b['current'].to(device);base=b['base_prediction'].to(device);feat=b['decoder_features'].to(device);target=b['target'].to(device)
   with torch.autocast('cuda',dtype=torch.bfloat16):
    out=c(cur,base,feat);loss=phase_a_objective(base+out['correction'],target,out['correction_regularizer_terms'],c.cell_area,c.std_dx)
   (loss/accum).backward();loss_sum+=float(loss.detach())*len(cur);seen+=len(cur)
   if (bi+1)%accum==0 or bi+1==len(loader):
    lr=learning_rate_for_update(global_step,total,warm,float(cfg['phase_a']['learning_rate']),float(cfg['phase_a']['minimum_learning_rate']))
    for g in opt.param_groups:g['lr']=lr
    torch.nn.utils.clip_grad_norm_(c.parameters(),float(cfg['phase_a']['gradient_clip_norm']));opt.step();opt.zero_grad(set_to_none=True);ema.update({'corrector':c});global_step+=1
  with ema.apply({'corrector':c}): metrics=eval_phase_a(c,val,stats,lat,device)
  score=float(metrics['mean_normalized_rmse']);ck=run/f'phase_a_epoch_{epoch:03d}.pt';atomic_torch_save(ck,{'epoch':epoch,'corrector_ema':ema.state_dict(),'optimizer':opt.state_dict(),'metrics':metrics,'score':score,'config':cfg,'predictor_checkpoint':str(predictor_path),'predictor_sha256':hashlib.sha256(predictor_path.read_bytes()).hexdigest(),'residual_stats_sha256':hashlib.sha256((ROOT/cfg['data'].get('residual_stats','results/predictor_residual_stats.npz')).read_bytes()).hexdigest()});records.append({'epoch':epoch,'checkpoint':str(ck),'score':score,'metrics':metrics,'train_loss':loss_sum/seen,'seconds':time.perf_counter()-t0});atomic_json_write(ledger,{'status':'running','epoch':epoch,'effective_batch':effective,'records':records})
 predictor_state=torch.load(predictor_path,map_location='cpu',weights_only=False);base6=predictor_state['validation_metrics'];selected=choose_phase_a_checkpoint(records,base6);final=run/'phase_a_best_ema.pt';os.replace(selected['checkpoint'],final)
 for p in run.glob('phase_a_epoch_*.pt'):p.unlink()
 selected['checkpoint']=str(final);atomic_json_write(run/'phase_a_selection.json',selected);atomic_json_write(ledger,phase_status_payload(cfg,'phase_a',status='completed',completed_epoch=epochs,effective_batch=effective,records=records,config_path=config_path,selected=selected));return final

def load_phase_a_corrector(cfg,stats,lat,res,device):
 c=build_corrector(cfg,stats,lat,res,device);p=phase_a_checkpoint_path(cfg,ROOT);s=torch.load(p,map_location='cpu',weights_only=False);ema=CorrectorEMA({'corrector':c});ema.load_state_dict(s['corrector_ema']);c.load_state_dict(ema_weights(ema,'corrector'),strict=True);return c,p

def phase_b(config_path):
 cfg,audit,stats,lat,res=load_payloads(config_path);device=torch.device('cuda');set_reproducibility(int(cfg['seed']));predictor,predictor_path=load_predictor(cfg,stats,lat,device);c,phasea=load_phase_a_corrector(cfg,stats,lat,res,device);set_phase_b_trainable(predictor,c);train=FourStepCorrectorDataset(CorrectorCacheDataset(ROOT/cfg['data']['cache_root']/'train'));val=CorrectorCacheDataset(ROOT/cfg['data']['cache_root']/'val');micro=int(cfg['phase_b']['micro_batch_size']);accum,effective=accumulation_plan(int(cfg['phase_b']['effective_global_batch_size']),micro);from .corrector_objectives import build_phase_b_optimizer;protocol=PhaseBProtocol(epochs=int(cfg['phase_b']['epochs']),corrector_learning_rate=float(cfg['phase_b']['corrector_learning_rate']),predictor_tail_learning_rate=float(cfg['phase_b']['predictor_tail_learning_rate']),betas=tuple(cfg['phase_b']['betas']),eps=float(cfg['phase_b']['eps']),weight_decay=float(cfg['phase_b']['weight_decay_matrix']),warmup_epochs=int(cfg['phase_b']['warmup_epochs']),minimum_lr_ratio=float(cfg['phase_b']['minimum_lr_ratio']),gradient_clip_norm=float(cfg['phase_b']['gradient_clip_norm']),ema_decay=float(cfg['phase_b']['ema_decay']),lead_weights=tuple(cfg['phase_b']['lead_weights']),channel_weights=tuple(float(x) for x in cfg['phase_b']['channel_weights']));opt=build_phase_b_optimizer(predictor,c,device=device,protocol=protocol);ema=CorrectorEMA({'predictor':predictor,'corrector':c},protocol.ema_decay);epochs=configured_epochs(cfg,'phase_b');updates=math.ceil(math.ceil(len(train)/micro)/accum);total=epochs*updates;warm=protocol.warmup_epochs*updates;run=ROOT/cfg['outputs']['run_dir'];top=[];ledger=run/'phase_b_status.json';global_step=0
 state_mean=torch.tensor(stats['state_mean'],device=device);state_std=torch.tensor(stats['state_std'],device=device);std_dx=torch.tensor(stats['delta_std'],device=device)
 for epoch in range(1,epochs+1):
  set_phase_b_trainable(predictor,c);loader=DataLoader(train,batch_size=micro,shuffle=True,generator=torch.Generator().manual_seed(1000+epoch),num_workers=0,drop_last=False);opt.zero_grad(set_to_none=True);loss_sum=0.;seen=0;t0=time.perf_counter()
  for bi,b in enumerate(loader):
   b={k:(v.to(device) if torch.is_tensor(v) else v) for k,v in b.items()}
   with torch.autocast('cuda',dtype=torch.bfloat16): loss,_=phase_b_rollout_objective(predictor,c,b['initial_x'],b['targets'],b['current_climatologies'],b['target_climatologies'],b['calendars'],state_mean=state_mean,state_std=state_std,std_dx=std_dx,area=c.cell_area,protocol=protocol)
   if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite Phase B loss epoch={epoch} batch={bi}')
   (loss/accum).backward();loss_sum+=float(loss.detach())*len(b['initial_x']);seen+=len(b['initial_x'])
   if (bi+1)%accum==0 or bi+1==len(loader):
    base_lr=learning_rate_for_update(global_step,total,warm,1.,.1)
    for g in opt.param_groups:g['lr']=g.get('initial_lr',g['lr'])*base_lr if 'initial_lr' in g else g['lr']*base_lr
    trainable=[p for p in list(predictor.parameters())+list(c.parameters()) if p.requires_grad]
    if not gradients_are_finite(trainable): raise FloatingPointError(f'nonfinite Phase B gradient epoch={epoch} batch={bi} global_step={global_step}')
    torch.nn.utils.clip_grad_norm_(trainable,protocol.gradient_clip_norm,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True);ema.update({'predictor':predictor,'corrector':c});global_step+=1
  with ema.apply({'predictor':predictor,'corrector':c}): ev=evaluate_corrector(predictor,c,val,stats,lat,split='val',leads=(6,24),batch_size=16,device=device)
  score=float(ev['metrics']['24']['mean_normalized_rmse']);ck=run/f'phase_b_epoch_{epoch:03d}.pt';atomic_torch_save(ck,{'epoch':epoch,'joint_ema':ema.state_dict(),'metrics_6_24':ev,'score':score,'config':cfg,'phase_a_checkpoint':str(phasea)});top.append({'epoch':epoch,'checkpoint':str(ck),'score':score});top.sort(key=lambda x:x['score']);
  for old in top[3:]: Path(old['checkpoint']).unlink(missing_ok=True)
  top=top[:3];atomic_json_write(ledger,{'status':'running','epoch':epoch,'effective_batch':effective,'top3':top,'train_loss':loss_sum/seen,'seconds':time.perf_counter()-t0})
 base=base_validation(cfg,stats,lat,device);candidates=[]
 for row in top:
  s=torch.load(row['checkpoint'],map_location='cpu',weights_only=False);ema.load_state_dict(s['joint_ema'])
  with ema.apply({'predictor':predictor,'corrector':c}): ev=evaluate_corrector(predictor,c,val,stats,lat,split='val',leads=FINAL_LEADS,batch_size=16,device=device)
  candidates.append({'checkpoint':row['checkpoint'],'metrics':ev['metrics'],'finite':True,'correction_stable':True})
 selected=choose_phase_b_checkpoint(candidates,base);final=run/'final_ema.pt';os.replace(selected['checkpoint'],final)
 for p in run.glob('phase_b_epoch_*.pt'):p.unlink()
 selected['checkpoint']=str(final);atomic_json_write(run/'phase_b_selection.json',selected);atomic_json_write(ledger,phase_status_payload(cfg,'phase_b',status='completed',completed_epoch=epochs,effective_batch=effective,records=top,config_path=config_path,selected=selected));return final

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--phase',choices=['a','b','all'],default='all');a=p.parse_args();
 if a.phase in ('a','all'):phase_a(a.config)
 if a.phase in ('b','all'):phase_b(a.config)
if __name__=='__main__':main()
