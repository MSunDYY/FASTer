from functools import partial
from pcdet.ops.iou3d_nms import iou3d_nms_utils

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict

    def double_flip(self, points):
        # y flip
        points_yflip = points.copy()
        points_yflip[:, 1] = -points_yflip[:, 1]

        # x flip
        points_xflip = points.copy()
        points_xflip[:, 0] = -points_xflip[:, 0]

        # x y flip
        points_xyflip = points.copy()
        points_xyflip[:, 0] = -points_xyflip[:, 0]
        points_xyflip[:, 1] = -points_xyflip[:, 1]

        return points_yflip, points_xflip, points_xyflip

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        if config.get('DOUBLE_FLIP', False):
            voxels_list, voxel_coords_list, voxel_num_points_list = [voxels], [coordinates], [num_points]
            points_yflip, points_xflip, points_xyflip = self.double_flip(points)
            points_list = [points_yflip, points_xflip, points_xyflip]
            keys = ['yflip', 'xflip', 'xyflip']
            for i, key in enumerate(keys):
                voxel_output = self.voxel_generator.generate(points_list[i])
                voxels, coordinates, num_points = voxel_output

                if not data_dict['use_lead_xyz']:
                    voxels = voxels[..., 3:]
                voxels_list.append(voxels)
                voxel_coords_list.append(coordinates)
                voxel_num_points_list.append(num_points)

            data_dict['voxels'] = voxels_list
            data_dict['voxel_coords'] = voxel_coords_list
            data_dict['voxel_num_points'] = voxel_num_points_list
        else:
            data_dict['voxels'] = voxels
            data_dict['voxel_coords'] = coordinates
            data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict
    
    def image_normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_normalize, config=config)
        mean = config.mean
        std = config.std
        compose = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )
        data_dict["camera_imgs"] = [compose(img) for img in data_dict["camera_imgs"]]
        return data_dict
    
    def image_calibrate(self,data_dict=None, config=None):
        if data_dict is None:
            return partial(self.image_calibrate, config=config)
        img_process_infos = data_dict['img_process_infos']
        transforms = []
        for img_process_info in img_process_infos:
            resize, crop, flip, rotate = img_process_info

            rotation = torch.eye(2)
            translation = torch.zeros(2)
            # post-homography transformation
            rotation *= resize
            translation -= torch.Tensor(crop[:2])
            if flip:
                A = torch.Tensor([[-1, 0], [0, 1]])
                b = torch.Tensor([crop[2] - crop[0], 0])
                rotation = A.matmul(rotation)
                translation = A.matmul(translation) + b
            theta = rotate / 180 * np.pi
            A = torch.Tensor(
                [
                    [np.cos(theta), np.sin(theta)],
                    [-np.sin(theta), np.cos(theta)],
                ]
            )
            b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
            b = A.matmul(-b) + b
            rotation = A.matmul(rotation)
            translation = A.matmul(translation) + b
            transform = torch.eye(4)
            transform[:2, :2] = rotation
            transform[:2, 3] = translation
            transforms.append(transform.numpy())
        data_dict["img_aug_matrix"] = transforms
        return data_dict

    def aug_roi_by_noise_torch(self, roi_boxes3d, gt_boxes3d, iou3d_src, aug_times=10, pos_thresh=None, config=None,
                               roi_labels=None):
        def random_aug_box3d(box3d, config):
            """
            :param box3d: (7) [x, y, z, h, w, l, ry]
            random shift, scale, orientation
            """

            if config.REG_AUG_METHOD == 'single':
                pos_shift = (torch.rand(3, device=box3d.device) - 0.5) * 0.8  # [-0.5 ~ 0.5]
                hwl_scale = (torch.rand(3, device=box3d.device) - 0.5) / (0.5 / 0.12) + 1.0  #
                angle_rot = (torch.rand(1, device=box3d.device) - 0.5) / (0.5 / (np.pi / 12))  # [-pi/12 ~ pi/12]
                aug_box3d = torch.cat(
                    [box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot, box3d[7:]], dim=0)
                return aug_box3d
            elif config.REG_AUG_METHOD == 'multiple':
                # pos_range, hwl_range, angle_range, mean_iou
                range_config = [[0.2, 0.1, np.pi / 12, 0.7],
                                [0.3, 0.15, np.pi / 12, 0.6],
                                [0.5, 0.15, np.pi / 9, 0.5],
                                [0.8, 0.15, np.pi / 6, 0.3],
                                [1.0, 0.15, np.pi / 3, 0.2]]
                idx = torch.randint(low=0, high=len(range_config), size=(1,))[0].long()

                pos_shift = ((torch.rand(3, device=box3d.device) - 0.5) / torch.tensor([0.5, 0.5, 1])) * \
                            range_config[idx][0]

                hwl_scale = ((torch.rand(3, device=box3d.device) - 0.5) / torch.tensor([0.5, 0.5, 1])) * \
                            range_config[idx][1] + 1.0

                angle_rot = ((torch.rand(1, device=box3d.device) - 0.5) / 0.5) * range_config[idx][2]

                aug_box3d = torch.cat(
                    [box3d[0:3] + pos_shift, box3d[3:6] * hwl_scale, box3d[6:7] + angle_rot, box3d[7:]], dim=0)
                return aug_box3d
            elif config.REG_AUG_METHOD == 'normal':
                x_shift = np.random.normal(loc=0, scale=0.3)
                y_shift = np.random.normal(loc=0, scale=0.2)
                z_shift = np.random.normal(loc=0, scale=0.3)
                h_shift = np.random.normal(loc=0, scale=0.25)
                w_shift = np.random.normal(loc=0, scale=0.15)
                l_shift = np.random.normal(loc=0, scale=0.5)
                ry_shift = ((torch.rand() - 0.5) / 0.5) * np.pi / 12

                aug_box3d = np.array([box3d[0] + x_shift, box3d[1] + y_shift, box3d[2] + z_shift, box3d[3] + h_shift,
                                      box3d[4] + w_shift, box3d[5] + l_shift, box3d[6] + ry_shift], dtype=np.float32)
                aug_box3d = torch.from_numpy(aug_box3d).type_as(box3d)
                return aug_box3d
            else:
                raise NotImplementedError

        iou_of_rois = torch.zeros(roi_boxes3d.shape[0]).type_as(gt_boxes3d)
        if pos_thresh is None:
            pos_thresh = torch.tensor(config.REG_FG_THRESH)

        for k in range(roi_boxes3d.shape[0]):
            temp_iou = cnt = 0
            roi_box3d = roi_boxes3d[k]

            gt_box3d = gt_boxes3d[k].view(1, gt_boxes3d.shape[-1])
            aug_box3d = roi_box3d
            keep = True
            while temp_iou < (pos_thresh) and cnt < aug_times:
                if np.random.rand() <= config.RATIO:
                    aug_box3d = roi_box3d  # p=RATIO to keep the original roi box
                    keep = True
                else:
                    aug_box3d = random_aug_box3d(roi_box3d, config=config)
                    keep = False
                aug_box3d = aug_box3d.view((1, aug_box3d.shape[-1]))
                iou3d = iou3d_nms_utils.boxes_iou3d_cpu(aug_box3d[:, :7], gt_box3d[:, :7])
                temp_iou = iou3d[0][0]
                cnt += 1
            roi_boxes3d[k] = aug_box3d.view(-1)
            if cnt == 0 or keep:
                iou_of_rois[k] = iou3d_src[k]
            else:
                iou_of_rois[k] = temp_iou
        return roi_boxes3d, iou_of_rois




    def anchor_aug(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.anchor_aug, config=config)

        def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
            """
            Args:
                rois: (N, 7)
                roi_labels: (N)
                gt_boxes: (N, )
                gt_labels:

            Returns:

            """
            """
            :param rois: (N, 7)
            :param roi_labels: (N)
            :param gt_boxes: (N, 8)
            :return:
            """
            max_overlaps = rois.new_zeros(rois.shape[0])
            gt_assignment = roi_labels.new_zeros(roi_labels.shape[0])

            for k in range(gt_labels.min().item(), gt_labels.max().item() + 1):
                roi_mask = (roi_labels == k)
                gt_mask = (gt_labels == k)
                if roi_mask.sum() > 0 and gt_mask.sum() > 0:
                    cur_roi = rois[roi_mask]
                    cur_gt = gt_boxes[gt_mask]
                    original_gt_assignment = gt_mask.nonzero().view(-1)

                    iou3d = iou3d_nms_utils.boxes_iou3d_cpu(cur_roi[:, :7], cur_gt[:, :7])  # (M, N)
                    cur_max_overlaps, cur_gt_assignment = torch.max(iou3d, dim=1)
                    # cur_gt_assignment = np.argmax(iou3d,axis=1)
                    max_overlaps[roi_mask] = cur_max_overlaps
                    gt_assignment[roi_mask] = original_gt_assignment[cur_gt_assignment]

            return max_overlaps, gt_assignment

        def generate_trajectory_msf(cur_batch_boxes, batch_dict,num_frames):


            trajectory_rois = cur_batch_boxes[None, :, :].repeat(num_frames, 1, 1)


            roi_list = batch_dict.get('roi_list', None)

            # batch_valid_length[0] = batch_dict['roi_scores'].squeeze(0)
            for i in range(1, num_frames):
                frame = trajectory_rois[i - 1].clone()
                frame[:, 0:2] = frame[:, 0:2] + (frame[:, 7:9] if batch_dict['sample_idx'] - i >= 0 else 0)

                if i % 1 == 0 and roi_list is not None:
                    iou3d = iou3d_nms_utils.boxes_iou3d_cpu(frame[:, :7], roi_list[i, :, :7])

                    max_overlaps, gt_assignment = iou3d.max(-1)

                    fg_inds = (max_overlaps > 0.5).nonzero().squeeze(-1)
                    frame[fg_inds] = roi_list[i, gt_assignment[fg_inds], :9]

                trajectory_rois[i, :, :] = frame


            return trajectory_rois

        def subsample_rois(max_overlaps, roi_labels, config):

            def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, hard_bg_ratio):
                if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
                    hard_bg_rois_num = min(int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
                    easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

                    # sampling hard bg
                    rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
                    hard_bg_inds = hard_bg_inds[rand_idx]

                    # sampling easy bg
                    rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
                    easy_bg_inds = easy_bg_inds[rand_idx]

                    bg_inds = torch.cat([hard_bg_inds, easy_bg_inds], dim=0)
                elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
                    hard_bg_rois_num = bg_rois_per_this_image
                    # sampling hard bg
                    rand_idx = torch.randint(low=0, high=hard_bg_inds.numel(), size=(hard_bg_rois_num,)).long()
                    bg_inds = hard_bg_inds[rand_idx]
                elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
                    easy_bg_rois_num = bg_rois_per_this_image
                    # sampling easy bg
                    rand_idx = torch.randint(low=0, high=easy_bg_inds.numel(), size=(easy_bg_rois_num,)).long()
                    bg_inds = easy_bg_inds[rand_idx]
                else:
                    raise NotImplementedError

                return bg_inds

            # sample fg, easy_bg, hard_bg
            fg_rois_per_image = int(np.round(config.FG_RATIO * config.ROI_PER_IMAGE))
            fg_thresh = min(config.REG_FG_THRESH, config.CLS_FG_THRESH)

            fg_inds = ((max_overlaps >= fg_thresh)).nonzero().view(-1)
            easy_bg_inds = ((max_overlaps < config.CLS_BG_THRESH_LO)).nonzero().view(-1)
            hard_bg_inds = ((max_overlaps < config.REG_FG_THRESH) &
                            (max_overlaps >= config.CLS_BG_THRESH_LO)).nonzero().view(-1)

            fg_num_rois = fg_inds.numel()
            bg_num_rois = hard_bg_inds.numel() + easy_bg_inds.numel()

            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(max_overlaps).long()
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                # sampling bg
                bg_rois_per_this_image = config.ROI_PER_IMAGE - fg_rois_per_this_image
                bg_inds = sample_bg_inds(
                    hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, config.HARD_BG_RATIO
                )

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                rand_num = np.floor(np.random.rand(config.ROI_PER_IMAGE) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(max_overlaps).long()
                fg_inds = fg_inds[rand_num]
                bg_inds = torch.tensor([]).type_as(fg_inds)

            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                bg_rois_per_this_image = config.ROI_PER_IMAGE
                bg_inds = sample_bg_inds(
                    hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, config.HARD_BG_RATIO
                )
            else:
                print('maxoverlaps:(min=%f, max=%f)' % (max_overlaps.min().item(), max_overlaps.max().item()))
                print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
                raise NotImplementedError

            sampled_inds = torch.cat((fg_inds, bg_inds), dim=0)
            return sampled_inds.long(), fg_inds.long(), bg_inds.long()

        def sample_rois_for_mppnet(batch_dict, config):

            cur_frame_idx = 0
            num_frame = config.NUM_FRAMES
            rois = batch_dict['trajectory_rois'][cur_frame_idx, :, :]

            gt_boxes = torch.from_numpy(batch_dict['gt_boxes']).float()

            code_size = rois.shape[-1]

            trajectory_rois = batch_dict['trajectory_rois']

            batch_valid_length = torch.zeros(
                ((batch_dict['trajectory_rois'].shape[0], config.ROI_PER_IMAGE)))

            cur_trajectory_rois = trajectory_rois

            cur_roi, cur_gt, cur_roi_labels = rois, gt_boxes, data_dict['roi_labels'].long()

            if 'valid_length' in batch_dict.keys():
                cur_valid_length = batch_dict['valid_length']

            k = cur_gt.__len__() - 1
            while k > 0 and cur_gt[k].sum() == 0:
                k -= 1

            cur_gt = cur_gt[:k + 1]
            cur_gt = torch.zeros(1, cur_gt.shape[1]) if len(cur_gt) == 0 else cur_gt

            if config.get('SAMPLE_ROI_BY_EACH_CLASS', False):
                max_overlaps, gt_assignment = get_max_iou_with_same_class(
                    rois=cur_roi, roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7], gt_labels=cur_gt[:, -1].long()
                )

            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_cpu(cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps, gt_assignment = torch.max(iou3d, dim=1)
            num_rois = max_overlaps.shape[0]
            sampled_inds, fg_inds, bg_inds = subsample_rois(
                max_overlaps=max_overlaps[torch.arange(num_rois)], roi_labels=cur_roi_labels, config=config)

            sampled_inds = torch.concat([fg_inds, bg_inds], dim=0)
            if config.get('USE_OVERLAP_ROI', False):
                iou3d = iou3d_nms_utils.boxes_iou3d_cpu(cur_roi[sampled_inds, :7], cur_roi[:, :7]).max(0)[0]
                sampled_inds = torch.nonzero(iou3d > 0).squeeze() if (iou3d > 0).sum() <= 180 else \
                torch.topk(iou3d, 180)[1]
                fg_inds = sampled_inds[max_overlaps[sampled_inds] >= config.REG_FG_THRESH]
                bg_inds = sampled_inds[max_overlaps[sampled_inds] < config.REG_FG_THRESH]
                sampled_inds = torch.concat([fg_inds, bg_inds], dim=0)

            batch_roi_labels = cur_roi_labels[sampled_inds.long()]

            if config.get('USE_ROI_AUG', False):

                fg_rois, fg_iou3d = self.aug_roi_by_noise_torch(cur_roi[fg_inds], cur_gt[gt_assignment[fg_inds]],
                                                                max_overlaps[fg_inds],
                                                                aug_times=config.ROI_FG_AUG_TIMES, config=config,
                                                                roi_labels=cur_roi_labels)

                bg_rois = cur_roi[bg_inds]
                bg_iou3d = max_overlaps[bg_inds]

                batch_rois = torch.cat([fg_rois, bg_rois], 0)
                batch_roi_ious = torch.cat([fg_iou3d, bg_iou3d], 0)
                batch_gt_of_rois = cur_gt[gt_assignment[sampled_inds]]

            else:
                batch_rois = cur_roi[sampled_inds]
                batch_roi_ious = max_overlaps[sampled_inds]
                batch_gt_of_rois = cur_gt[gt_assignment[sampled_inds]]

                # batch_roi_scores[index] = cur_roi_scores[sampled_inds]

            if 'valid_length' in batch_dict.keys():
                batch_valid_length = cur_valid_length[:, sampled_inds]

            if config.USE_TRAJ_AUG.ENABLED:

                batch_trajectory_rois_list = [batch_rois.unsqueeze(0)]
                # batch_trajectory_rois_list = []
                for idx in range(1, num_frame):
                    if idx == cur_frame_idx:
                        batch_trajectory_rois_list.append(
                            cur_trajectory_rois[cur_frame_idx:cur_frame_idx + 1, sampled_inds])

                        continue
                    fg_backs, _ = self.aug_roi_by_noise_torch(cur_trajectory_rois[idx, fg_inds],
                                                              cur_trajectory_rois[idx, fg_inds][:, :8],
                                                              max_overlaps[fg_inds], \
                                                              aug_times=config.ROI_FG_AUG_TIMES,
                                                              pos_thresh=config.USE_TRAJ_AUG.THRESHOD, config=config)
                    bg_backs = cur_trajectory_rois[idx, bg_inds]

                    batch_trajectory_rois_list.append(torch.cat([fg_backs, bg_backs], 0)[None, :, :])

                batch_trajectory_rois = torch.cat(batch_trajectory_rois_list, 0)

            else:
                batch_trajectory_rois = cur_trajectory_rois[:, sampled_inds]
                # batch_trajectory_rois[index] = cur_trajectory_rois[:,sampled_inds]

            return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_labels, batch_trajectory_rois, batch_valid_length



        with torch.no_grad():
            data_dict['roi_boxes'] = torch.from_numpy(data_dict['roi_boxes'])
            data_dict['roi_labels'] = torch.from_numpy(data_dict['roi_labels'])
            data_dict['roi_scores'] = torch.from_numpy(data_dict['roi_scores'])
            data_dict['roi_list'] = data_dict['roi_boxes'].clone()
            if data_dict['roi_boxes'].shape[0] > 1:
                # data_dict['roi_list'] = torch.concat(
                #     [data_dict['roi_boxes'].clone(), data_dict['roi_scores'].unsqueeze(-1).clone(),
                #      data_dict['roi_labels'].unsqueeze(-1).clone()], -1)
                mask = data_dict['roi_boxes'][0,:,0] != 0

                data_dict['roi_boxes'] = data_dict['roi_boxes'][0, mask]

                data_dict['roi_scores'] = data_dict['roi_scores'][0, mask]
                data_dict['roi_labels'] = data_dict['roi_labels'][0, mask]

            data_dict['num_rois'] = torch.tensor(data_dict['roi_scores'].shape[0])

            trajectory_rois = generate_trajectory_msf(
                data_dict['roi_boxes'].reshape(-1, data_dict['roi_boxes'].shape[-1]),
                data_dict,config.NUM_FRAMES)
            data_dict['trajectory_rois'] = trajectory_rois
            data_dict.pop('roi_list')
            data_dict['has_class_labels'] = True
            if not self.training:
                return data_dict

            batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_labels, batch_trajectory_rois, batch_valid_length = sample_rois_for_mppnet(
                batch_dict=data_dict, config=config)

            REG_FG_THRESH = torch.tensor([0.55, 0.4, 0.4])
            CLS_FG_THRESH = torch.tensor([0.75, 0.6, 0.6])
            CLS_BG_THRESH = torch.tensor([0.35, 0.2, 0.2])
            reg_valid_mask = (batch_roi_ious > REG_FG_THRESH[batch_roi_labels - 1]).long()
            if config.CLS_SCORE_TYPE == 'cls':
                batch_cls_labels = (batch_roi_ious > config.CLS_FG_THRESH).long()
                ignore_mask = (batch_roi_ious > config.CLS_BG_THRESH) & \
                              (batch_roi_ious < config.CLS_FG_THRESH)
                batch_cls_labels[ignore_mask > 0] = -1
            elif config.CLS_SCORE_TYPE == 'roi_iou':
                # iou_bg_thresh = config.CLS_BG_THRESH
                # iou_fg_thresh = config.CLS_FG_THRESH
                iou_bg_thresh = CLS_BG_THRESH[batch_roi_labels - 1]
                iou_fg_thresh = CLS_FG_THRESH[batch_roi_labels - 1]

                fg_mask = batch_roi_ious > iou_fg_thresh
                bg_mask = batch_roi_ious < iou_bg_thresh
                interval_mask = (fg_mask == 0) & (bg_mask == 0)

                batch_cls_labels = (fg_mask > 0).float()
                batch_cls_labels[interval_mask] = \
                    (batch_roi_ious[interval_mask] - iou_bg_thresh[interval_mask]) / (
                                iou_fg_thresh[interval_mask] - iou_bg_thresh[interval_mask])
            else:
                raise NotImplementedError

            targets_dict = {'rois': batch_rois, 'gt_of_rois': batch_gt_of_rois,
                            'gt_iou_of_rois': batch_roi_ious,  # 'roi_scores': batch_roi_scores,
                            'roi_labels': batch_roi_labels, 'reg_valid_mask': reg_valid_mask,
                            'rcnn_cls_labels': batch_cls_labels,
                            'trajectory_rois': batch_trajectory_rois,
                            'valid_length': batch_valid_length,
                            'num_rois': torch.tensor([batch_rois.shape[0]]).long(),
                            'stack_concat': True if config.get('STACK_CONCAT', False) else False
                            }

            rois = targets_dict['rois']  # (B, N, 7 + C)
            gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
            targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

            # canonical transformation
            roi_center = rois[:, 0:3]
            roi_ry = rois[:, 6] % (2 * np.pi)
            gt_of_rois[:, 0:3] = gt_of_rois[:, 0:3] - roi_center
            gt_of_rois[:, 6] = gt_of_rois[:, 6] - roi_ry

            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
            ).view(-1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = gt_of_rois[:, 6] % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
            heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (
                        2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

            gt_of_rois[:, 6] = heading_label
            targets_dict['gt_of_rois'] = gt_of_rois
            data_dict['targets_dict'] = targets_dict
            poped_key = [ 'trajectory_rois']

            for key in poped_key:
                data_dict.pop(key)
            return data_dict
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
