"""Held-out 2017-2018 full-window evaluation with ClimODE metrics."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np,torch
from ..data.cache import CachedWeatherDataset
from .metrics import evaluate_forecast,LEADS_HOURS
from ..models.predictor import StructuredPredictor


def valid_test_starts(dataset,steps=24):
    years=dataset.arrays['year']
    return [i for i in range(len(dataset)-steps+1) if years[i]==years[i+steps-1]]

def yearly_climatologies(dataset):
    years=np.asarray(dataset.arrays['year']);result={}
    for year in np.unique(years):
        idx=np.flatnonzero(years==year)
        states=np.asarray(dataset.arrays['current'][idx],dtype=np.float64)
        final=np.asarray(dataset.arrays['target'][idx[-1]],dtype=np.float64)[None]
        result[int(year)]=np.concatenate((states,final)).mean(0).astype(np.float32)
    return result

def model_input_batch(current,climate,mean,std,calendar):
    anomaly=(current-climate)/std[None,:,None,None];climate_norm=(climate-mean[None,:,None,None])/std[None,:,None,None]
    day=calendar[:,2].float();slot=calendar[:,3].float()
    phase=torch.stack((torch.sin(2*math.pi*day/365),torch.cos(2*math.pi*day/365),torch.sin(2*math.pi*slot/4),torch.cos(2*math.pi*slot/4)),1)
    return torch.cat((anomaly,climate_norm,phase[:,:,None,None].expand(-1,-1,32,64)),1)

def _checkpoint_weights(state):
    if isinstance(state, dict):
        ema = state.get('ema')
        if isinstance(ema, dict) and isinstance(ema.get('shadow'), dict): return ema['shadow']
        for key in ('model','model_state_dict','state_dict'):
            if isinstance(state.get(key), dict): return state[key]
    return state

def evaluate_test(config_path,checkpoint,output,batch_size=32,leads=LEADS_HOURS):
    leads=tuple(int(h) for h in leads)
    if not leads or any(h <= 0 or h % 6 for h in leads): raise ValueError('leads must be positive six-hour multiples')
    cfg=json.loads(Path(config_path).read_text());root=Path(__file__).parents[2];audit=json.loads((root/'results/data_audit.json').read_text());stats=audit['normalization'];lat=np.asarray(audit['grid']['latitude'],dtype=np.float32);device='cuda'
    ds=CachedWeatherDataset(root/'cache/era5/test');starts=valid_test_starts(ds,max(leads)//6);clims=yearly_climatologies(ds)
    model=StructuredPredictor(torch.tensor(lat).deg2rad(),stats['state_std'],stats['delta_std']).to(device).eval();state=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(_checkpoint_weights(state),strict=True)
    mean=torch.tensor(stats['state_mean'],device=device);std=torch.tensor(stats['state_std'],device=device)
    keep={h:([],[],[]) for h in leads}
    with torch.inference_mode():
      for left in range(0,len(starts),batch_size):
        index=np.asarray(starts[left:left+batch_size]);current=torch.from_numpy(np.asarray(ds.arrays['current'][index])).to(device);cc=torch.from_numpy(np.asarray(ds.arrays['current_climatology'][index])).to(device)
        years=np.asarray(ds.arrays['year'][index]);annual=torch.from_numpy(np.stack([clims[int(y)] for y in years]))
        for step in range(1,max(leads)//6+1):
          rows=index+step-1;tc=torch.from_numpy(np.asarray(ds.arrays['target_climatology'][rows])).to(device);calendar=torch.from_numpy(np.asarray(ds.arrays['calendar'][rows])).to(device)
          current=model(model_input_batch(current,cc,mean,std,calendar),current,cc,tc);cc=tc
          hour=step*6
          if hour in keep:
            keep[hour][0].append(current.cpu());keep[hour][1].append(torch.from_numpy(np.asarray(ds.arrays['target'][rows])).clone());keep[hour][2].append(annual)
    metrics={}
    for hour,(pred,truth,clim) in keep.items():
      metrics[str(hour)]=evaluate_forecast(torch.cat(pred),torch.cat(truth),torch.cat(clim),torch.tensor(lat),torch.tensor(stats['state_std']))
    result={'status':'test','variant':'full','checkpoint':str(checkpoint),'split':'test','years':[2017,2018],'n_windows':len(starts),'protocol':'ClimODE physical-unit cosine-latitude RMSE and per-year anomaly ACC; sample-first aggregation','metrics':metrics}
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--leads',default=','.join(map(str,LEADS_HOURS)));a=p.parse_args();leads=tuple(int(x) for x in a.leads.split(','));print(json.dumps(evaluate_test(a.config,a.checkpoint,a.output,a.batch_size,leads),allow_nan=False))
if __name__=='__main__':main()
