from os import getgrouplist
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F
from typing import Optional, List
from torch import Tensor
from torch.nn.init import xavier_uniform_, zeros_, kaiming_normal_
from spconv.pytorch.utils import PointToVoxel

import math
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from torch.nn.functional import linear
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from pcdet import device
# from torch.utils.cpp_extension import load
# scatter = load(name='scatter', sources=['../cuda/scatter.cpp', '../cuda/scatter.cu'])

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads,dropout = 0.0, ln=False):
        super(Attention, self).__init__()
        self.dim_LIN = dim
        self.num_heads = num_heads
        self.fc = nn.Linear(dim,dim*3)
        self.dropout = nn.Dropout(dropout)
        if ln:
            self.ln0 = nn.LayerNorm(dim)
            self.ln1 = nn.LayerNorm(dim)
        self.fc_o = nn.Linear(dim, dim)

    def forward(self, Q, drop=True):
        B = Q.shape[0]

        Q,K, V = self.fc(Q).chunk(3,-1)
        dim_split = self.dim_LIN // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        Q_ = Q_/math.sqrt(dim_split)
        A = torch.softmax(Q_.bmm(K_.transpose(1,2)), 2)
        A = self.dropout(A)
        if self.num_heads >= 2:
            temp = A.split(Q.size(0),dim=0)
            temp = torch.stack([tensor_ for tensor_ in temp], dim=0)
            weight = torch.mean(temp, dim=0)

        if drop:
            sampled_inds = torch.topk(weight.sum(1),A.shape[-1]//2,1)[1]
            sampled_inds_ = sampled_inds.repeat(self.num_heads,1)

            O = torch.gather(A, 1,sampled_inds_[:,:,None].repeat(1, 1,A.shape[-1])).bmm(V_)
            O = torch.concat(O.chunk(self.num_heads,0),-1)
            return self.fc_o(O), weight, sampled_inds
        else:
            O =torch.concat( A.bmm(V_).chunk(self.num_heads,0),dim=-1)
            return self.fc_o(O),weight,None

class Adaptive_MHSA(nn.Module):
    def __init__(self, dim, num_heads,dropout = 0.0, ln=False,batch_first = True):
        super(Adaptive_MHSA, self).__init__()
        self.dim_LIN = dim
        self.num_heads = num_heads
        self.in_proj_weight = Parameter(torch.empty((3 * dim, dim)))
        self.in_proj_bias = Parameter(torch.empty(3 * dim))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = NonDynamicallyQuantizableLinear(dim, dim, bias=True)
        self._reset_parameters()
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias,0)

        nn.init.constant_(self.out_proj.bias,0)

    def forward(self, Q,K,V, drop=0.5):
        B,T,D = Q.shape
        L_out = int(T*drop)
        Q,K, V = linear(Q,self.in_proj_weight,self.in_proj_bias).chunk(3,-1)
        dim_split = self.dim_LIN // self.num_heads
        Q = Q.view(B,T,self.num_heads,dim_split).transpose(1,2).contiguous().view(B*self.num_heads,T,dim_split)
        K = K.view(B,T,self.num_heads,dim_split).transpose(1,2).contiguous().view(B*self.num_heads,T,dim_split)
        V = V.view(B,T,self.num_heads,dim_split).transpose(1,2).contiguous().view(B*self.num_heads,T,dim_split)
        Q = Q / math.sqrt(dim_split)
        A = torch.softmax(torch.bmm(Q,K.transpose(1,2)), 2)
        A = self.dropout(A)

        if drop !=1:

            weight = A.view(B,self.num_heads,T,T).max(1)[0]
            # var = weight.transpose(1,2).flatten(0,1).var(0)
            weight = weight .sum(1)
            # V = (V.unflatten(0,(B,self.num_heads)) * weight.unsqueeze(-1)).flatten(0,1)
            sampled_inds = torch.topk(weight,int(weight.shape[-1]*drop),-1)[1]
            # sampled_inds = torch.arange(int(weight.shape[-1]*drop),device=device)[None,:].repeat(weight.shape[0],1)
            A = torch.gather(A.unflatten(0,(-1,self.num_heads)),2,sampled_inds[:,None,:,None].repeat(1,self.num_heads,1,T)).flatten(0,1)
        else:
            weight = None
            sampled_inds = None
        O = torch.bmm(A,V)
        O = O.view(B,self.num_heads,L_out,dim_split).transpose(1,2).contiguous().view(B,L_out,self.dim_LIN)
        O = linear(O,self.out_proj.weight,self.out_proj.bias)
        return O,weight,sampled_inds


class SpatialMixerBlock(nn.Module):

    def __init__(self, channels,num_heads=8, dropout=0.0,batch_first=False):
        super().__init__()

        self.mixer = nn.MultiheadAttention(channels, num_heads, dropout=dropout,batch_first=batch_first)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
                               nn.Linear(channels, 2*channels),
                               nn.ReLU(),
                               nn.Dropout(dropout),
                               nn.Linear(2*channels, channels),
                               )
    def forward(self, src,return_weight=True):
       
        src2,weight = self.mixer(src, src,src)
        
        src = src + self.dropout(src2)
        src_mixer = self.norm(src)
        sampled_index = torch.topk(weight.sum(1),weight.shape[-1]//2,-1)[1]
        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)
        if return_weight:
            return src_mixer,weight,sampled_index
        else:
            return src_mixer


class SpatialDropBlock(nn.Module):

    def __init__(self, channels, config=None, dropout=0.0, batch_first=False):
        super().__init__()

        self.mixer = Adaptive_MHSA(channels,8,dropout,batch_first= True)
        # self.mixer = Attention(channels, 8,dropout=dropout )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, src, return_weight=False,drop=0.5):


        src2,weight,sampled_inds = self.mixer(src,src,src,drop=drop)
        if drop!=1:
            src =torch.gather(src,1,sampled_inds[:,:,None].repeat(1,1,src.shape[-1]))

        src = src+self.dropout(src2)
        src_mixer = self.norm(src)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)
        if return_weight:
            return src_mixer, weight,sampled_inds
        else:
            return src_mixer


class CrossMixerBlock(nn.Module):

    def __init__(self, channels, config=None, dropout=0.0,batch_first = False):
        super().__init__()

        self.mixer = nn.MultiheadAttention(channels, 8, dropout=dropout,batch_first=batch_first)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

        self.norm_channel = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * channels, channels),
        )
    def emb_pos(self,tensor,pos):
        return tensor if pos is None else tensor+pos
    def forward(self, query,key,pos_q=None,pos_k=None,return_weight=False):
        
        query = query if pos_q is None else query+pos_q
       
        src2, weight = self.mixer(self.emb_pos(query,pos_q), self.emb_pos(key,pos_k), key)

        query = query + self.dropout(src2)
        src_mixer = self.norm(query)

        src_mixer = src_mixer + self.ffn(src_mixer)
        src_mixer = self.norm_channel(src_mixer)
        if return_weight:
            return src_mixer, weight
        else:
            return src_mixer
class Transformer(nn.Module):

    def __init__(self, config, d_model=512, nhead=8, num_encoder_layers=6,
                dim_feedforward=2048, dropout=0.1,activation="relu", normalize_before=False,
                num_lidar_points=None, share_head=True,num_groups=None,
                sequence_stride=None,num_frames=None):
        super().__init__()

        self.config = config
        self.share_head = share_head
        self.num_frames = num_frames
        self.nhead = nhead
        self.sequence_stride = sequence_stride
        self.num_groups = num_groups
        self.num_lidar_points = num_lidar_points
        self.d_model = d_model
        self.nhead = nhead
        encoder_layer = [TransformerEncoderLayer(self.config, d_model, nhead, dim_feedforward,dropout, activation, 
                      normalize_before, num_lidar_points,num_groups=num_groups) for i in range(num_encoder_layers)]

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,self.config)
        self.token = nn.Parameter(torch.zeros(self.num_groups, 1, d_model))

        
        if self.num_frames >4:
  
            self.group_length = self.num_frames // self.num_groups
            self.fusion_all_group = MLP(input_dim = self.config.hidden_dim*self.group_length, 
               hidden_dim = self.config.hidden_dim, output_dim = self.config.hidden_dim, num_layers = 4)

            self.fusion_norm = FFN(d_model, dim_feedforward)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,batch_dict, pos=None):

        BS, N, C = src.shape
        if not pos is None:
            pos = pos.permute(1, 0, 2)
            

        token_list = [self.token[i:(i+1)].repeat(BS,1,1) for i in range(self.num_groups)]

        # src = [torch.cat([torch.gather(token_list[i],1,batch_dict['roi_labels'].view(-1,1,1).repeat(1,1,self.d_model)-1,),src[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points]],dim=1) for i in range(self.num_groups)]
        src = [torch.concat([token_list[i],src[:,i*self.num_lidar_points:(i+1)*self.num_lidar_points]],dim=1) for i in range(self.num_groups)]
        src = torch.cat(src,dim=0)

        # src = src.permute(1, 0, 2)
        memory,tokens = self.encoder(src,batch_dict,pos=pos)

        # memory = torch.cat(memory[0:1].chunk(self.num_groups,dim=1),0)
        return memory[:,:1], tokens,memory[:,1:]
    

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None,config=None):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer)
        self.num_layers = num_layers
        self.norm = norm
        self.config = config

    def forward(self, src,batch_dict,
                pos: Optional[Tensor] = None):

        token_list = []
        output = src
        for layer in self.layers:
            output,tokens = layer(output,batch_dict,pos=pos)
            token_list.append(tokens)
        if self.norm is not None:
            output = self.norm(output)

        return output,token_list
class vector_attention(nn.Module):
    def __init__(self,d_model,nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.linear_q = nn.ModuleList()
        self.linear_w = nn.ModuleList()
        self.linear_k = nn.ModuleList()
        self.linear_v = nn.ModuleList()
        for i in range(nhead):

            self.linear_w.append(nn.Sequential(nn.BatchNorm1d(d_model),nn.ReLU(inplace=True),
                                          nn.Linear(d_model,d_model),
                                          nn.BatchNorm1d(d_model),nn.ReLU(inplace=True),
                                          nn.Linear(d_model,d_model)))
            self.softmax = nn.Softmax(dim=0)
            self.linear_q.append(nn.Linear(d_model,d_model))
            self.linear_k.append(nn.Linear(d_model, d_model))
            self.linear_v.append(nn.Linear(d_model, d_model))
            self.MLP = MLP(d_model*nhead,hidden_dim=d_model,output_dim=d_model,num_layers=3)
    def forward(self,q,k,v,pos=None):
        x_list = []

        for i in range(self.nhead):
            q,k,v = self.linear_q[i](q),self.linear_k[i](k),self.linear_v[i](v)
            w = k -q
            for j,layer in enumerate(self.linear_w[i]): w =layer(w.permute(1,2,0).contiguous()).permute(2,0,1).contiguous() if j%3==0 else layer(w)
            w = self.softmax(w)
            x = (w*v).sum(0).unsqueeze(0)
            x_list.append(x)
        # w_all = w_all/w_all.sum(0)[None,:]
        x =self.MLP(torch.concat(x_list,dim=-1))
        return x

class TransformerEncoderLayer(nn.Module):
    count = 0
    def __init__(self, config, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,num_points=None,num_groups=None):
        super().__init__()
        TransformerEncoderLayer.count += 1
        self.layer_count = TransformerEncoderLayer.count
        self.config = config
        self.num_point = num_points
        self.num_groups= num_groups
        self.point_feature = 96
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,batch_first=True)
        # self.self_attn = vector_attention(d_model, nhead=4)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)



        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.mlp_mixer_3d = SpatialDropBlock(
                        self.config.hidden_dim,
                        self.config.use_mlp_mixer,
                        dropout=dropout,
        )

        if self.layer_count<=self.config.enc_layers-1 and config.get('sampler',False) is not False:
            from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_utils

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,batch_dict,
                     pos: Optional[Tensor] = None):


        src_intra_group_fusion,weight,sampled_inds = self.mlp_mixer_3d(src[:,1:],return_weight=True,drop = self.config.drop_rate[self.layer_count-1])
        if False:
            points = batch_dict['points'][batch_dict['points'][:, -1] == 0, 1:]
            box = batch_dict['gt_boxes'][0,4:]
            dis = (((points[:, :2] - box[:,:2]) ** 2).sum(-1)) / ((box[:,3:5] / 2) ** 2).sum(-1)

            points_color = np.ones(((batch_dict['points'][:, -1] == 0).sum().item(), 3))
            points_color[dis.cpu().numpy() < 1.1] = np.array([0.5, 0.5, 0])
            draw_scenes(points, gt_boxes=box, point_colors=points_color,background_color = np.ones(3)*240/255)



        token = src[:,:1]

        if not pos is None:
            key = self.with_pos_embed(src_intra_group_fusion, pos[1:])
        else:
            key = src_intra_group_fusion


        src_summary = self.self_attn(token, key,src_intra_group_fusion)[0]
        token = token + self.dropout1(src_summary)
        token = self.norm1(token)
        src_summary = self.linear2(self.dropout(self.activation(self.linear1(token))))
        token = token + self.dropout2(src_summary)
        token = self.norm2(token)

        src = torch.cat([token,src[:,1:]],1)

        src = torch.cat([src[:, :1], src_intra_group_fusion], 1)

        if self.config.drop_rate[self.layer_count-1]<1:

            batch_dict['src_idx'] = torch.gather(batch_dict['src_idx'],1,sampled_inds)

        return src, torch.cat(src[:,:1].chunk(self.num_groups,0),1)

    def forward_pre(self, src,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,batch_dict,
                pos: Optional[Tensor] = None):

        if self.normalize_before:
            return self.forward_pre(src, pos)
        return self.forward_post(src,batch_dict,  pos)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,dout=None,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward(self, tgt,tgt_input):
        tgt = tgt + self.dropout2(tgt_input)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

def build_transformer(args):
    return Transformer(
        config = args,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        num_lidar_points = args.num_lidar_points,
        num_frames = args.num_frames,
        sequence_stride = args.get('sequence_stride',1),
        num_groups=args.num_groups,
    )


class VoxelSampler(nn.Module):
    GAMMA = 1.05

    def __init__(self, device, voxel_size, pc_range, max_points_per_voxel, num_point_features=5):
        super().__init__()

        self.voxel_size = voxel_size

        self.gen = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, pc_range[5] - pc_range[2]],
            coors_range_xyz=pc_range,
            num_point_features=num_point_features,
            max_num_voxels=50000,
            max_num_points_per_voxel=max_points_per_voxel,
            device=device
        )

        self.pc_start = torch.FloatTensor(pc_range[:2]).to(device)
        self.k = max_points_per_voxel
        self.grid_x = int((pc_range[3] - pc_range[0]) / voxel_size)
        self.grid_y = int((pc_range[4] - pc_range[1]) / voxel_size)

    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_size, trajectory_rois, num_sample, points_list , num_rois = None):

        src = list()
        num_frames = trajectory_rois.shape[0]
        speed = torch.norm(trajectory_rois[..., -2:], dim=-1)
        for bs_idx in range(batch_size):

            # cur_points = batch_dict['points'][(batch_dict['points'][:, 0] == bs_idx)][:, 1:]
            cur_points = points_list[bs_idx * num_frames:(bs_idx+1) * num_frames]
            cur_batch_boxes = trajectory_rois[:,num_rois[bs_idx]:num_rois[bs_idx+1]]
            src_points = list()
            for idx in range(num_frames):

                cur_time_points = cur_points[idx]
                gamma = torch.clamp((self.GAMMA * (1 + speed[idx,num_rois[bs_idx]:num_rois[bs_idx+1]])) ** (idx / 5), max=2)  # ** (idx+1)

                cur_frame_boxes = cur_batch_boxes[idx]
                if idx==0:
                    src_points.append(torch.zeros(cur_batch_boxes.shape[1],num_sample,cur_time_points.shape[-1],device=device))
                    continue
                    voxel, coords, num_points = self.gen(cur_time_points)
                    coords = coords[:, [2, 1]].contiguous()

                    query_coords = (cur_frame_boxes[:, :2] - self.pc_start) // self.voxel_size

                    radiis = torch.ceil(
                    torch.norm(cur_frame_boxes[:, 3:5] / 2, dim=-1) * gamma  / self.voxel_size)

                    dist = torch.abs(query_coords[:, None, :2] - coords[None, :, :])

                    voxel_mask = torch.all(dist < radiis[:, None, None], dim=-1).any(0)

                    num_points = num_points[voxel_mask]
                    key_points = voxel[voxel_mask, :]

                    point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)

                    point_mask = num_points[:, None] > point_mask
                    key_points = key_points[point_mask]
                    key_points = key_points[torch.randperm(len(key_points)), :]
                else:
                    key_points = cur_time_points
                key_points = self.cylindrical_pool(key_points, cur_frame_boxes, num_sample, gamma)

                src_points.append(key_points)

            src.append(torch.stack(src_points))

        return torch.concat(src,1).transpose(0,1)

    def cylindrical_pool(self, cur_points, cur_boxes, num_sample, gamma=1.):
        if len(cur_points) < num_sample:
            cur_points = F.pad(cur_points, [0, 0, 0, num_sample - len(cur_points)])

        cur_radiis = torch.norm(cur_boxes[:, 3:5] / 2, dim=-1) * gamma
        dis = torch.norm(
            (cur_points[:, :2].unsqueeze(0) - cur_boxes[:, :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1)), dim=2)
        point_mask = (dis <= cur_radiis.unsqueeze(-1))

        sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample)

        sampled_idx = sampled_idx.view(-1, 1).repeat(1, cur_points.shape[-1])
        sampled_points = torch.gather(cur_points, 0, sampled_idx).view(len(sampled_mask), num_sample, -1)

        sampled_points[sampled_mask == 0, :] = 0

        return sampled_points


class VoxelPointsSampler(nn.Module):
    GAMMA = 1.1

    def __init__(self, device, voxel_size, pc_range, max_points_per_voxel, num_point_features=5,):
        super().__init__()

        self.voxel_size = voxel_size

        self.gen = PointToVoxel(
            vsize_xyz=[voxel_size, voxel_size, pc_range[5] - pc_range[2]],
            coors_range_xyz=pc_range,
            num_point_features=num_point_features,
            max_num_voxels=50000,
            max_num_points_per_voxel=max_points_per_voxel,
            device=device
        )
        self.num_recall_points=0
        self.num_gt_points=0
        self.num_recall_new = 0
        self.num_points = 0
        self.iteration = 0

        self.pc_start = torch.FloatTensor(pc_range[:2]).to(device)
        self.k = max_points_per_voxel
        self.grid_x = int((pc_range[3] - pc_range[0]) / voxel_size)
        self.grid_y = int((pc_range[4] - pc_range[1]) / voxel_size)


    def get_output_feature_dim(self):
        return self.num_point_features

    def forward(self, batch_size, trajectory_rois, num_sample, batch_dict,  num_rois=None, return_raw_points = False):
        src = list()
        query_points_list = list()
        src_idx_list = list()
        points_pre_list = list()

        gamma = self.GAMMA

        cur_time_points = batch_dict['points'][batch_dict['points'][:, -1] == 0][:, :-1]

        for bs_idx in range(batch_size):
            cur_batch_points = cur_time_points[cur_time_points[:, 0] == bs_idx, 1:].contiguous()
            cur_batch_boxes = trajectory_rois[
                                  num_rois[bs_idx]:num_rois[bs_idx + 1]]

            voxel, coords, num_points = self.gen(cur_batch_points)
            num_voxel=num_points
            coords = coords[:, [2, 1]].contiguous()
            query_coords = (cur_batch_boxes[:, :2] - self.pc_start) / self.voxel_size

            radiis = torch.norm(cur_batch_boxes[:, 3:5] / 2, dim=-1)*gamma/ self.voxel_size

            dist = torch.norm(query_coords[:, None, :2] - coords[None, :, :],dim=-1)
            voxel_mask = (dist < radiis[:, None]).any(0)

            if self.training and batch_dict.get('pred_key_boxes',None) is not None:
                pre_roi = batch_dict['pred_key_boxes'][bs_idx]

                pre_roi = pre_roi.flatten(0,1)
                pre_roi = pre_roi[pre_roi[:,2]!=0]
                query_coords_pre = (pre_roi[:,:2] - self.pc_start) / self.voxel_size
                radiis_pre = torch.norm(pre_roi[:,3:5]/2,dim=-1)/self.voxel_size
                dist_pre = torch.norm(query_coords_pre[:,None,:2] - coords[None,:,:],dim=-1)
                voxel_mask_pre = (dist_pre<radiis_pre[:,None]).any(0)
                voxel_mask_pre = voxel_mask_pre*(~voxel_mask)
                key_points_pre = voxel[voxel_mask_pre,:]
                num_points_pre = num_points[voxel_mask_pre]
                point_mask_pre = torch.arange(self.k,device=device)[None,:].repeat(len(key_points_pre),1)
                point_mask_pre = point_mask_pre<num_points_pre[:,None]
                key_points_pre = key_points_pre[point_mask_pre]
                if key_points_pre.shape[0]>0:
                    points_in_boxes_idx = roiaware_pool3d_utils.points_in_boxes_gpu(key_points_pre[None, :, :3],
                                                                      pre_roi[None,:,:7]).squeeze(0)
                    key_points_pre = key_points_pre[points_in_boxes_idx>=0]

                points_pre_list.append(key_points_pre)
            else:
                pre_roi = None
            num_points = num_points[voxel_mask]
            key_points = voxel[voxel_mask, :]

            point_mask = torch.arange(self.k)[None, :].repeat(len(key_points), 1).type_as(num_points)

            point_mask = num_points[:, None] > point_mask
            key_points = key_points[point_mask]
            key_points = key_points[torch.randperm(len(key_points)), :]

            sampled_points, src_idx ,query_points = self.cylindrical_pool(key_points, cur_batch_boxes,
                                                        num_sample, gamma, pre_roi=None if self.training else pre_roi)
            if return_raw_points:
                query_points = key_points
            src.append(sampled_points)
            src_idx_list.append(src_idx)
            query_points_list.append(query_points)
            # src.append(torch.stack(src_points))
        return torch.concat(src, dim=0),torch.concat(src_idx_list,dim=0),query_points_list,points_pre_list if (not self.training and pre_roi is not None) else None

    def cylindrical_pool(self, cur_points, cur_boxes, num_sample, gamma=1.,pre_roi=None):
        if len(cur_points) < num_sample:
            cur_points = F.pad(cur_points, [0, 0, 0, num_sample - len(cur_points)])

        cur_radiis = torch.norm(cur_boxes[:, 3:5] / 2, dim=-1) * gamma
        dis = torch.norm(
            (cur_points[:, :2].unsqueeze(0) - cur_boxes[:, :2].unsqueeze(1).repeat(1, cur_points.shape[0], 1)), dim=2)



        point_mask = (dis <= cur_radiis.unsqueeze(-1))

        sampled_mask, sampled_idx = torch.topk(point_mask.float(), num_sample, 1)

        sampled, idx = torch.unique(sampled_idx, return_inverse=True)
        query_points = cur_points[sampled]


        sampled_mask = sampled_mask.bool()
        sampled_idx = sampled_idx.view(-1, 1).repeat(1, cur_points.shape[-1])
        sampled_points = torch.gather(cur_points, 0, sampled_idx).view(len(sampled_mask), num_sample, -1)
        idx = idx * sampled_mask

        # points_pre = cur_points[~point_mask.sum(0).bool()]
        sampled_points = sampled_points * sampled_mask[:,:,None]
        return sampled_points, idx,query_points
def build_voxel_sampler(device, return_idx=False):
    if not return_idx:
        return VoxelSampler(
            device,
            voxel_size=0.4,
            pc_range=[-75.2, -75.2, -10, 75.2, 75.2, 10],
            max_points_per_voxel=32,
            num_point_features=5,

        )
    else:
        return VoxelPointsSampler(
            device,
            voxel_size=0.4,
            pc_range=[-75.2, -75.2, -10, 75.2, 75.2, 10],
            max_points_per_voxel=32,
            num_point_features=5,
        )



