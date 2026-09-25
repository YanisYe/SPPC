"""End-to-end S2-Hodge forecast: neural heads plus FP32 physics."""
from __future__ import annotations
import torch
from torch import nn
from .backbone import SphericalBackbone
from .physics import (GridAdapter,HodgeToroidalProjection,GeostrophicWind,
                      SemiLagrangianTransport,SharedEdgeDivergence,unit_sphere_cell_areas)

class StructuredPredictor(nn.Module):
    def __init__(self,latitudes,state_std,delta_std,aux_wind_std=None):
        super().__init__();self.neural=SphericalBackbone()
        lat=torch.as_tensor(latitudes,dtype=torch.float32);self.grid=self.neural.grid_adapter
        self.hodge=HodgeToroidalProjection(32,64,16,16);self.geo=GeostrophicWind(lat,64)
        self.transport=SemiLagrangianTransport(lat,64);areas=unit_sphere_cell_areas(lat.double(),64).float();self.divergence=SharedEdgeDivergence(areas)
        self.register_buffer('areas',areas);self.register_buffer('state_std',torch.as_tensor(state_std,dtype=torch.float32));self.register_buffer('delta_std',torch.as_tensor(delta_std,dtype=torch.float32))
        aux=torch.ones(4) if aux_wind_std is None else torch.as_tensor(aux_wind_std,dtype=torch.float32);self.register_buffer('aux_std',aux)
    def project_era_wind(self,wind):
        with torch.autocast(device_type=wind.device.type, enabled=False):
            return self.grid.vector_to_era(self.hodge(self.grid.vector_to_official(wind.float())))
    def forward(self,model_input,current,current_climate,target_climate,return_parts=False):
        raw=self.neural(model_input); current=current.float();cc=current_climate.float();tc=target_climate.float(); anomaly=current-cc
        surface=current[:,3:5]+3*self.delta_std[3:5,None,None]*torch.tanh(raw['raw_surface_residual'].float())
        mid10=.5*(current[:,3:5]+surface)
        wind_logits=raw['raw_wind'].float()
        wind850=self.project_era_wind(3*self.aux_std[None,:2,None,None]*torch.tanh(wind_logits[:,:2]))
        ug,vg=self.geo(current[:,2]);geo=self.project_era_wind(torch.stack((ug,vg),1))
        correction=self.project_era_wind(3*self.aux_std[None,2:,None,None]*torch.tanh(wind_logits[:,2:]))
        w500=self.project_era_wind(geo+correction)
        predicted_transport=torch.cat([self.transport(anomaly[:,i:i+1],*wind) for i,wind in enumerate(((mid10[:,0],mid10[:,1]),(wind850[:,0],wind850[:,1]),(w500[:,0],w500[:,1])))],1)
        transported=predicted_transport
        east=torch.tanh(raw['raw_edge_east'].float())*self.delta_std[:3,None,None]
        north=torch.tanh(raw['raw_edge_north'].float())*self.delta_std[:3,None,None]
        correction=self.divergence(east,north)
        source=3*self.delta_std[:3]*torch.tanh(raw['raw_global_source'].float())
        scalar=tc[:,:3]+transported+correction+source[:,:,None,None]
        prediction=torch.cat((scalar,surface),1)
        if not return_parts:return prediction
        return prediction,{'wind850':wind850,'wind500':w500,'mid10':mid10,'transported':transported,'correction':correction,'source':source,'raw':raw}
