import torch
from .detector3d_template import Detector3DTemplate
from pcdet.ops.iou3d_nms import iou3d_nms_utils
import os
import numpy as np
import time
from ...utils import common_utils
from ..model_utils import model_nms_utils
from pcdet import device


class FASTer(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.module_list[0].signal='val'

    def simple_refer(self,batch_dict,epoch):
        self.roi_head.simple_refer(batch_dict,epoch)

    def forward(self, batch_dict):
        for cur_module in self.module_list[:]:
            batch_dict = cur_module(batch_dict)
        
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }

            return ret_dict, tb_dict, disp_dict
        else:
            if 'batch_box_preds' in batch_dict.keys():
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # print(recall_dicts, torch.count_nonzero(pred_dicts[0]['pred_labels'] == 2), torch.count_nonzero(batch_dict['gt_boxes'][0, :, 7] == 2))
                return pred_dicts, recall_dicts
            else:
                return None,None
    def get_training_loss(self):
        disp_dict = {}  
        tb_dict ={}
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
        loss = loss_rcnn

        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):

            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                try:
                    cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                except:
                    record_dict = {
                        'pred_boxes': torch.tensor([]),
                        'pred_scores': torch.tensor([]),
                        'pred_labels': torch.tensor([])
                    }
                    pred_dicts.append(record_dict)
                    continue

                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                
                # selected, selected_scores = model_nms_utils.class_agnostic_nms(
                #     box_scores=cls_preds, box_preds=box_preds,
                #     nms_config=post_process_cfg.NMS_CONFIG,
                #     score_thresh=post_process_cfg.SCORE_THRESH
                # )

                selected, selected_scores = model_nms_utils.class_specific_nms(
                    box_scores=cls_preds, box_preds=box_preds,box_labels=label_preds-1,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

                #########  Car DONOT Using NMS ###### 
                if post_process_cfg.get('NOT_APPLY_NMS_FOR_VEL',False):
                    
                    pedcyc_mask = final_labels !=1 
                    final_scores_pedcyc = final_scores[pedcyc_mask]
                    final_labels_pedcyc = final_labels[pedcyc_mask]
                    final_boxes_pedcyc = final_boxes[pedcyc_mask]

                    car_mask = (label_preds==1) & (cls_preds > post_process_cfg.SCORE_THRESH)
                    final_scores_car = cls_preds[car_mask]
                    final_labels_car = label_preds[car_mask]
                    final_boxes_car = box_preds[car_mask]

                    final_scores  = torch.cat([final_scores_car,final_scores_pedcyc],0)
                    final_labels  = torch.cat([final_labels_car,final_labels_pedcyc],0)
                    final_boxes  = torch.cat([final_boxes_car,final_boxes_pedcyc],0)

                #########  Car DONOT Using NMS ###### 
            batch_dict['rois'] = batch_dict['roi_boxes'].unsqueeze(0)
            batch_dict['final_scores'] = final_scores
            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
            # cur_gt = batch_dict['gt_boxes'][0]
            # if cur_gt.shape[0] > 0:
            #     iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7]).to(device)
            #     recall_gt_threshold = torch.full((cur_gt.shape[0],), fill_value=0.7, device=device)
            #     recall_gt_threshold[cur_gt[:, -1] > 1] = 0.5
            #     max_overlaps, gt_assignment = torch.max(iou3d_rcnn, dim=-1)
            #     fg_roi = (max_overlaps >= recall_gt_threshold[gt_assignment]).nonzero().view(-1)
            #     unique, indice = torch.unique(gt_assignment[fg_roi], return_inverse=True)
            #     mask = torch.zeros_like(fg_roi, dtype=torch.bool)
            #     for i in unique:
            #         mask[(gt_assignment[fg_roi] == i).nonzero()[0]] = True
            #     fg_roi_ind = fg_roi[mask]
            #     fg_roi = torch.zeros(box_preds.shape[0], dtype=torch.bool, device=device)
            #     fg_roi[fg_roi_ind] = True
            #     # final_scores = cls_preds
            #     # print((-torch.log(final_scores[fg_roi])).max().item(),'  ' ,(-torch.log(1-final_scores[~fg_roi])).max().item())
            #     loss_pos_cls = (1 - cls_preds[fg_roi]).sum().item()
            #     loss_neg_cls = cls_preds[~fg_roi].sum().item()
            #     recall_dict['pred'][0] += fg_roi.sum().item()
            #     recall_dict['pred'][1] += (~fg_roi).sum().item()
            # else:
            #     loss_pos_cls = 0
            #     loss_neg_cls = cls_preds.sum().item()
            #     recall_dict['pred'][0] += 0
            #     recall_dict['pred'][1] += cls_preds.shape[0]
            # recall_dict['loss_cls'][0] += loss_pos_cls
            # recall_dict['loss_cls'][1] += loss_neg_cls


            record_dict = {
                'pred_boxes': final_boxes[:,:7],
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

