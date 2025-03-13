import os.path
from typing import ValuesView
import torch.nn as nn
import torch
import numpy as np
import copy
import torch.nn.functional as F

from ...utils import common_utils, loss_utils
from .roi_head_template import RoIHeadTemplate
from ..model_utils.faster_utils import build_transformer,CrossMixerBlock,SpatialMixerBlock,SpatialDropBlock
from ..model_utils.faster_utils import build_voxel_sampler,MLP


from pathlib import Path
import os

from pcdet import device





class Intra_Group_Fusion(nn.Module):
    def __init__(self,num_channels,num_groups_in,num_groups_out,):
        super(Intra_Group_Fusion, self).__init__()
        self.groups_in = num_groups_in
        self.groups_out = num_groups_out
        self.channels = num_channels
        self.conv = nn.Sequential(
            nn.Conv1d(self.channels * self.groups_in // self.groups_out,self.channels,1,1),
            nn.BatchNorm1d(self.channels),
            nn.ReLU()
        )
        self.linear = nn.ModuleList([nn.Linear(self.channels*2,self.channels) for _ in range(self.groups_in//self.groups_out)])
        self.dropout = nn.ModuleList([nn.Dropout(0.1) for _ in range(self.groups_in//self.groups_out)])
        self.norm = nn.LayerNorm(self.channels)
    def forward(self,src):
        src = src.unflatten(1, (-1, self.groups_out)).transpose(1, 2).flatten(0, 1)
        src_max = src.max(2).values
        src_max = src_max.flatten(1, 2)
        src_max = self.conv(src_max.unsqueeze(-1)).squeeze()
        src_new = [self.dropout[i](
            self.linear[i](torch.concat([src[:, i], src_max[:, None, :].repeat(1, src.shape[2], 1)], dim=-1))) for i in
                   range(self.groups_in // self.groups_out)]

        src = self.norm(src + torch.stack(src_new, 1)).flatten(1, 2)
        return src

class Grouped_Hierarchical_Fusion(nn.Module):
    def __init__(self,model_cfg=None):
        super(Grouped_Hierarchical_Fusion, self).__init__()
        self.channels = model_cfg.hidden_dim
        self.num_frames = model_cfg.num_frames
        self.num_groups = model_cfg.num_groups
        self.drop_rate = model_cfg.drop_rate
        self.attention = SpatialDropBlock(self.channels,dropout=0.1,batch_first=True)
        self.group_fusion1 = Intra_Group_Fusion(self.channels,self.num_groups[0],self.num_groups[1])

        self.attention2 = SpatialDropBlock(self.channels,dropout=0.1,batch_first=True)
        self.group_fusion2 = Intra_Group_Fusion(self.channels,self.num_groups[1],self.num_groups[2])

        self.attention3 = SpatialMixerBlock(self.channels,dropout=0.1,batch_first=True)

        self.decoder_layer = CrossMixerBlock(self.channels,dropout=0.1,batch_first=True)

    def forward(self, src,token,src_cur,batch_dict):
        B = src_cur.shape[0]


        token_list = list()
        src = src.reshape(src.shape[0]*self.num_frames,-1,src.shape[-1])

        src,weight,sampled_inds = self.attention(src,return_weight=True,drop=self.drop_rate[0])
        src = src.reshape(B,self.num_frames,-1,self.channels)

        src = self.group_fusion1(src)
        src,weight,sampled_inds = self.attention2(src,return_weight=True,drop=self.drop_rate[1])

        src = src.unflatten(0,(B,self.num_groups[1]))

        src = self.group_fusion2(src)
        src,weight,sampled_inds = self.attention3(src,return_weight = True)

        token = self.decoder_layer(token,src)
        token_list.append(token)

        return token_list

class FASTerHead(RoIHeadTemplate):
    def __init__(self,model_cfg, num_class=1,**kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg


        self.num_lidar_points = self.model_cfg.TransformerSSP.num_lidar_points
        self.num_lidar_points2 = self.model_cfg.TransformerMSP.num_lidar_points
        self.avg_stage1_score = self.model_cfg.get('AVG_STAGE1_SCORE', None)

        self.hidden_dim = model_cfg.TRANS_INPUT

        self.num_groups = model_cfg.TransformerSSP.num_groups
        self.num_frames = model_cfg.TransformerMSP.num_frames
        self.voxel_sampler_cur = build_voxel_sampler(device, return_idx = True)
        self.voxel_sampler = build_voxel_sampler(device,return_idx = False)

        self.jointembed = MLP(self.hidden_dim, model_cfg.TransformerSSP.hidden_dim, self.box_coder.code_size * self.num_class, 4)

        self.up_dimension_geometry = MLP(input_dim = 29, hidden_dim = 64, output_dim =self.hidden_dim, num_layers = 3)
        self.up_dimension_motion = MLP(input_dim = 30, hidden_dim = 64, output_dim =self.hidden_dim, num_layers = 3)

        self.transformerSSP = build_transformer(model_cfg.TransformerSSP)
        self.transformerMSP = Grouped_Hierarchical_Fusion(model_cfg.TransformerMSP)

        self.class_embed = nn.ModuleList()
        self.class_embed_final = nn.Linear(model_cfg.TransformerMSP.hidden_dim,1)
        self.class_embed.append(nn.Linear(model_cfg.TransformerSSP.hidden_dim, 1))

        self.token_root = Path('../../data/waymo/focal_tokens')
        self.bbox_embed = nn.ModuleList()
        for _ in range(self.num_groups):
            self.bbox_embed.append(MLP(model_cfg.TransformerSSP.hidden_dim, model_cfg.TransformerSSP.hidden_dim, self.box_coder.code_size * self.num_class, 4))

        self.token_bank = list()




    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.bbox_embed.layers[-1].weight, mean=0, std=0.001)

    def get_corner_points_of_roi(self, rois,with_center=False):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_corner_points(rois, batch_size_rcnn)
        local_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()

        global_roi_grid_points = local_roi_grid_points + global_center.unsqueeze(dim=1)
        if with_center:
            global_roi_grid_points = torch.concat([global_roi_grid_points,global_center[:,None,:]],dim=1)
        return global_roi_grid_points, local_roi_grid_points


    @staticmethod
    def get_corner_points(rois, batch_size_rcnn):
        faked_features = rois.new_ones((2, 2, 2))

        dense_idx = faked_features.nonzero()  
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = dense_idx * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  
        return roi_grid_points


    def spherical_coordinate(self, src, diag_dist):
        assert (src.shape[-1] == 27)
        device = src.device
        indices_x = torch.LongTensor([0,3,6,9,12,15,18,21,24]).to(device)  #
        indices_y = torch.LongTensor([1,4,7,10,13,16,19,22,25]).to(device) # 
        indices_z = torch.LongTensor([2,5,8,11,14,17,20,23,26]).to(device) 
        src_x = torch.index_select(src, -1, indices_x)
        src_y = torch.index_select(src, -1, indices_y)
        src_z = torch.index_select(src, -1, indices_z)
        dis = (src_x ** 2 + src_y ** 2 + src_z ** 2) ** 0.5
        phi = torch.atan(src_y / (src_x + 1e-5))
        the = torch.acos(src_z / (dis + 1e-5))
        dis = dis / (diag_dist + 1e-5)
        src = torch.cat([dis, phi, the], dim = -1)
        return src

    def get_proposal_aware_motion_feature(self, proxy_point,  trajectory_rois):
        num_rois = proxy_point.shape[0]
        padding_mask = proxy_point[:,:,0:1]!=0
        time_stamp = torch.ones([proxy_point.shape[0], proxy_point.shape[1], 1],device = device)
        padding_zero = torch.zeros([proxy_point.shape[0], proxy_point.shape[1], 2],device = device)
        point_time_padding = torch.cat([padding_zero, time_stamp], -1)

        num_frames = trajectory_rois.shape[1]
        num_points_single_frame = proxy_point.shape[1]//num_frames
        for i in range(num_frames):
            point_time_padding[:, i * num_points_single_frame : (i+1) * num_points_single_frame, -1] = i * 0.1
        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois.contiguous())

        corner_points = corner_points.flatten(-2,-1)
        trajectory_roi_center = trajectory_rois.flatten(0,1)[:, :3]
        corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)

        lwh = trajectory_rois.flatten(0,1)[:, 3:6].unsqueeze(1)
        diag_dist = (lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5

        if True:
            motion_aware_feat = proxy_point[:,:,:3].repeat(1,1,9)-corner_add_center_points.unflatten(0,(num_rois,-1))[:,0:1]
        else:
            motion_aware_feat = corner_add_center_points.unflatten(0,(num_rois,-1))-corner_add_center_points.unflatten(0,(num_rois,-1))[:,0:1]
            motion_aware_feat = motion_aware_feat.unsqueeze(2).repeat(1,1,num_points_single_frame,1).flatten(1,2)
        geometry_aware_feat = proxy_point.reshape(num_rois*num_frames,num_points_single_frame,-1)[:, :, :3].repeat(1, 1, 9) - corner_add_center_points.unsqueeze(1)
        motion_aware_feat = self.spherical_coordinate(motion_aware_feat, diag_dist=diag_dist.unflatten(0,(num_rois,-1))[:,:1,:])
        geometry_aware_feat = self.spherical_coordinate(geometry_aware_feat,diag_dist=diag_dist[:,:,None])

        # motion_aware_feat = self.up_dimension_motion(torch.cat([motion_aware_feat, point_time_padding,valid.transpose(0,1)[:,:,None].repeat(1,1,num_points_single_frame).reshape(num_rois,-1,1)], -1))
        motion_aware_feat = self.up_dimension_motion(torch.cat([motion_aware_feat, point_time_padding], -1))

        geometry_aware_feat = self.up_dimension_geometry(torch.concat([geometry_aware_feat.reshape(num_rois,num_frames*num_points_single_frame,-1),proxy_point[:,:,3:]],-1))

        return motion_aware_feat + geometry_aware_feat

    def get_proposal_aware_geometry_feature(self, src, trajectory_rois):
        padding_mask = src[:,:,0:1]!=0

        num_rois = trajectory_rois.shape[0]

        corner_points, _ = self.get_corner_points_of_roi(trajectory_rois.contiguous())

        # corner_points = corner_points.view(batch_size, num_rois, -1, corner_points.shape[-1])
        corner_points = corner_points.view(num_rois, -1)
        trajectory_roi_center = trajectory_rois[:, :3]
        corner_add_center_points = torch.cat([corner_points, trajectory_roi_center], dim=-1)
        proposal_aware_feat = src[:,:,:3].repeat(1, 1,9) - corner_add_center_points.unsqueeze(1).repeat(1, src.shape[1], 1)

        lwh = trajectory_rois[:, 3:6].unsqueeze(1).repeat(1,proposal_aware_feat.shape[1], 1)
        diag_dist = (lwh[:, :, 0] ** 2 + lwh[:, :, 1] ** 2 + lwh[:, :, 2] ** 2) ** 0.5
        proposal_aware_feat = self.spherical_coordinate(proposal_aware_feat, diag_dist=diag_dist.unsqueeze(-1))

        proposal_aware_feat = torch.cat([proposal_aware_feat, src[:, :, 3:]], dim=-1)

        src_gemoetry = self.up_dimension_geometry(proposal_aware_feat)

        return src_gemoetry
    
    def encode_torch(self,gt_boxes,points):
        gt_boxes[:, 3:6] = torch.clamp_min(gt_boxes[:, 3:6], min=1e-5)
        from pcdet.utils.common_utils import rotate_points_along_z
        xg, yg, zg, dxg, dyg, dzg, rg, *cgs = torch.split(gt_boxes, 1, dim=-1)
        
        points = rotate_points_along_z((points-gt_boxes[:,:3])[:,None,:],angle=-1 * gt_boxes[:,6]).squeeze()
        xa, ya, za = torch.split(points, 1, dim=-1)
        xt = xa/dxg
        yt = ya/dyg
        zt = za/dzg
        

        cts = [g for g in cgs]
        return torch.cat([xt, yt, zt, torch.cos(rg), torch.sin(rg), *cts], dim=-1)

       
    def pos_offset_encoding(self,src,boxes):
        radiis = torch.norm(boxes[:,3:5]/2,dim=-1)
        return (src-boxes[None,:,:3])/radiis[None,:,None].repeat(1,1,3)

    def update_tokens(self,raw_points,src_idx,frame_id):
        focal_points = raw_points[torch.unique(src_idx)]
        if frame_id==0:
            self.token_bank = [focal_points for _ in range(self.num_frames)]
        else:
            self.token_bank.pop(-1)
            self.token_bank = [focal_points] + self.token_bank

    def get_points_online(self, poses, ):
        focal_points_list = list()
        for i in range(1, self.num_frames):
            pose_pre2cur = torch.matmul(torch.inverse(poses[:4, :]), poses[4 * i:4 * (i + 1), :])
            expanded_points = torch.concat([self.token_bank[i][:, :3], torch.ones_like(self.token_bank[i][:, :1])],
                                           dim=-1)
            focal_points_list.append(
                torch.concat([torch.matmul(expanded_points, pose_pre2cur.T)[:, :3], self.token_bank[i][:, 3:]], dim=-1))
        return focal_points_list

    def simple_refer(self,batch_dict,epoch_id):
        batch_size = batch_dict['batch_size']
        num_sample = self.num_lidar_points
        trajectory_rois = batch_dict['roi_boxes']
        num_rois = torch.cumsum(F.pad(batch_dict['num_rois'], (1, 0), 'constant', 0), 0).long()
        src_cur,src_idx,query_points,points_pre = self.voxel_sampler_cur(batch_size,trajectory_rois,num_sample,batch_dict,num_rois=num_rois,return_raw_points = epoch_id==0)

        batch_dict['src_cur'] = src_cur
        batch_dict['src_idx'] = src_idx
        batch_dict['query_points'] = query_points

        src_cur = self.get_proposal_aware_geometry_feature(src_cur, trajectory_rois)

        hs, tokens, src_cur = self.transformerSSP(src_cur, batch_dict, pos=None)
        src_idx = batch_dict['src_idx']
        for b in range(batch_size):
            src_idx_b = src_idx[num_rois[b]:num_rois[b+1]]
            focal_points = query_points[b][torch.unique(src_idx_b)] if epoch_id!=0 else query_points[b]
            focal_token_file = self.token_root/ batch_dict['frame_id'][b][:-4]/('%04d.npy'%batch_dict['sample_idx'])
            os.makedirs(self.token_root/ batch_dict['frame_id'][b][:-4],exist_ok=True)
            np.save(focal_token_file,focal_points.cpu().numpy())
    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        num_frames = self.num_frames
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_dict['cur_frame_idx'] = 0

        if self.training :
            targets_dict = batch_dict['targets_dict']
            trajectory_rois = targets_dict['trajectory_rois']



            num_rois = torch.cumsum(F.pad(targets_dict['num_rois'],(1,0),'constant',0),0)
        else:
            trajectory_rois = batch_dict['trajectory_rois']

            num_rois = torch.tensor([0,batch_dict['roi_boxes'].shape[0]],device=device)




        roi_boxes = batch_dict['roi_boxes']
        num_sample = self.num_lidar_points

        roi_boxes = roi_boxes.reshape(-1,roi_boxes.shape[-1])

        src_cur,src_idx,query_points,points_pre = self.voxel_sampler_cur(batch_size,trajectory_rois[0,...],num_sample,batch_dict,num_rois=num_rois)
        batch_dict['src_cur'] = src_cur
        batch_dict['src_idx'] = src_idx
        batch_dict['query_points'] = query_points
        # if self.model_cfg.get('USE_TRAJ_EMPTY_MASK', None):
        #     src_cur[empty_mask.view(-1)] = 0
        src_cur = self.get_proposal_aware_geometry_feature(src_cur,trajectory_rois[0,...])
        hs, tokens,src_cur = self.transformerSSP(src_cur, batch_dict, pos=None)
        if not self.training:
            self.update_tokens(query_points[0],batch_dict['src_idx'],batch_dict['sample_idx'].item())
            points_pre_list = self.get_points_online(batch_dict['poses'][0])
            points_pre_list = [points[:,1:-1]] + points_pre_list
        else:
            points_pre_list = [points[torch.logical_and(points[:,0]==b,points[:,-1]==f*0.1),1:-1] for b in range(batch_size) for f in range(num_frames)]

        src_pre_points = self.voxel_sampler(batch_size,trajectory_rois,self.num_lidar_points2,points_pre_list,num_rois)
        trajectory_rois = trajectory_rois.transpose(0,1)
        src_pre_points = src_pre_points.flatten(1,2)
        src_idx = batch_dict['src_idx'][:,:self.num_lidar_points2]
        for b in range(batch_size):
            src_cur_points = torch.gather(query_points[b],0,src_idx[num_rois[b]:num_rois[b+1]].reshape(-1,1).repeat(1,5)).reshape(-1,self.num_lidar_points2,5)
            src_pre_points[num_rois[b]:num_rois[b+1],:self.num_lidar_points2] = src_cur_points

        src_pre = self.get_proposal_aware_motion_feature(src_pre_points, trajectory_rois)

        tokens2 = self.transformerMSP(src_pre,tokens[-1],src_cur,batch_dict)

        point_cls_list = []
        point_reg_list = []

        for i in range(len(tokens)):
            point_cls_list.append(self.class_embed[0](tokens[i][:,0]))
        for i in range(len(tokens2)):
            point_cls_list.append(self.class_embed_final(tokens2[i][:,-1]))

        for j in range(len(tokens)):
            point_reg_list.append(self.bbox_embed[0](tokens[j][:,0]))
        point_cls = torch.cat(point_cls_list,0)
        point_reg = torch.cat(point_reg_list,0)
        joint_reg = self.jointembed(tokens2[-1].flatten(1,2))
        rcnn_cls = point_cls
        rcnn_reg = joint_reg
        if not self.training:
            rcnn_cls = rcnn_cls[-tokens2[-1].shape[0]:]

            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['roi_boxes'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            batch_dict['batch_box_preds'] = batch_box_preds

            batch_dict['cls_preds_normalized'] = False
            if self.avg_stage1_score:
                stage1_score = batch_dict['roi_scores'][:,None]
                batch_cls_preds = F.sigmoid(batch_cls_preds)
                if self.model_cfg.get('IOU_WEIGHT', None):
                    batch_box_preds_list = []
                    roi_labels_list = []
                    batch_cls_preds_list = []
                    # for bs_idx in range(batch_size):

                    car_mask = batch_dict['roi_labels'] ==1
                    batch_cls_preds_car = batch_cls_preds.pow(self.model_cfg.IOU_WEIGHT[0])* \
                                          stage1_score.pow(1-self.model_cfg.IOU_WEIGHT[0])
                    batch_cls_preds_car = batch_cls_preds_car[car_mask][None]
                    batch_cls_preds_pedcyc = batch_cls_preds.pow(self.model_cfg.IOU_WEIGHT[1])* \
                                             stage1_score.pow(1-self.model_cfg.IOU_WEIGHT[1])
                    batch_cls_preds_pedcyc = batch_cls_preds_pedcyc[~car_mask][None]
                    cls_preds = torch.cat([batch_cls_preds_car,batch_cls_preds_pedcyc],1)
                    box_preds = torch.cat([batch_dict['batch_box_preds'][car_mask],
                                                 batch_dict['batch_box_preds'][~car_mask]],0)[None]
                    roi_labels = torch.cat([batch_dict['roi_labels'][car_mask],
                                            batch_dict['roi_labels'][~car_mask]],0)[None]
                    batch_box_preds_list.append(box_preds)
                    roi_labels_list.append(roi_labels)
                    batch_cls_preds_list.append(cls_preds)


                    batch_dict['batch_box_preds'] = torch.cat(batch_box_preds_list,0)
                    batch_dict['roi_labels'] = torch.cat(roi_labels_list,0)
                    batch_cls_preds = torch.cat(batch_cls_preds_list,0)
                    
                else:
                    batch_cls_preds = torch.sqrt(batch_cls_preds*stage1_score)
                batch_dict['cls_preds_normalized']  = True

            batch_dict['batch_cls_preds'] = batch_cls_preds
        else:
            targets_dict['batch_size'] = batch_size
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['box_reg'] = rcnn_reg
            targets_dict['point_reg'] = point_reg
            targets_dict['point_cls'] = point_cls
            self.forward_ret_dict = targets_dict

        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict
    
    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)

        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)

        rcnn_reg = forward_ret_dict['rcnn_reg'] 

        roi_boxes3d = forward_ret_dict['rois']

        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}
        
        
        if loss_cfgs.REG_LOSS == 'smooth-l1':

            rois_anchor = roi_boxes3d.clone().detach()[:,:7].contiguous().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
                )
            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][0]

            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()
  
            if self.model_cfg.USE_AUX_LOSS:
                point_reg = forward_ret_dict['point_reg']

                groups = point_reg.shape[0]//reg_targets.shape[0]
                if groups != 1 :
                    point_loss_regs = 0
                    slice = reg_targets.shape[0]
                    for i in range(groups):
                        point_loss_reg = self.reg_loss_func(
                        point_reg[i*slice:(i+1)*slice].view(slice, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),) 
                        point_loss_reg = (point_loss_reg.view(slice, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                        point_loss_reg = point_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][2]
                        
                        point_loss_regs += point_loss_reg
                    point_loss_regs = point_loss_regs / groups
                    tb_dict['point_loss_reg'] = point_loss_regs.item()
                    rcnn_loss_reg += point_loss_regs 

                else:
                    point_loss_reg = self.reg_loss_func(point_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)  
                    point_loss_reg = (point_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                    point_loss_reg = point_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][2]
                    tb_dict['point_loss_reg'] = point_loss_reg.item()
                    rcnn_loss_reg += point_loss_reg

                seqbox_reg = forward_ret_dict['box_reg']  
                seqbox_loss_reg = self.reg_loss_func(seqbox_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),reg_targets.unsqueeze(dim=0),)
                seqbox_loss_reg = (seqbox_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
                seqbox_loss_reg = seqbox_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']*loss_cfgs.LOSS_WEIGHTS['traj_reg_weight'][1]
                tb_dict['seqbox_loss_reg'] = seqbox_loss_reg.item()
                rcnn_loss_reg += seqbox_loss_reg

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:

                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d[:,:7].contiguous().view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view( -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:,  6].view(-1)
                roi_xyz = fg_roi_boxes3d[:,  0:3].view(-1, 3)
                batch_anchors[:,  0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                corner_loss_func = loss_utils.get_corner_loss_lidar

                loss_corner = corner_loss_func(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7])

                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()

        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)

        
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':

            rcnn_cls_flat = rcnn_cls.view(-1)

            groups = rcnn_cls_flat.shape[0] // rcnn_cls_labels.shape[0]
            if groups != 1:
                rcnn_loss_cls = 0
                slice = rcnn_cls_labels.shape[0]
                for i in range(groups):
                    batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat[i*slice:(i+1)*slice]), 
                                     rcnn_cls_labels.float(), reduction='none')

                    cls_valid_mask = (rcnn_cls_labels >= 0).float() 
                    rcnn_loss_cls = rcnn_loss_cls + (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

                rcnn_loss_cls = rcnn_loss_cls / groups


            else:
                batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
                cls_valid_mask = (rcnn_cls_labels >= 0).float() 
                rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)

        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight'] 


        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}

        return rcnn_loss_cls, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds=None, box_preds=None):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)
        Returns:
        """
        code_size = self.box_coder.code_size
        if cls_preds is not None:
            batch_cls_preds = cls_preds.view( -1, cls_preds.shape[-1])
        else:
            batch_cls_preds = None
        batch_box_preds = box_preds.view( -1, code_size)

        roi_ry = rois[:,  6].view(-1)
        roi_xyz = rois[:,  0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:,  0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)

        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view( -1, code_size)
        batch_box_preds = torch.cat([batch_box_preds,rois[:,7:]],-1)
        return batch_cls_preds, batch_box_preds