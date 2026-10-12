import copy
import numpy as np
import os
from os import path as osp
import torch
import random
import json, pickle
import tempfile
import cv2
from pyquaternion import Quaternion
from mmcv.datasets import DATASETS
from mmcv.utils import track_iter_progress, mkdir_or_exist
from mmcv.parallel import DataContainer as DC
from mmcv.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from mmcv.fileio.io import load, dump
from mmcv.utils import save_tensor
from mmcv.datasets.pipelines import to_tensor
from mmcv.datasets.pipelines import Compose
from mmcv.datasets.custom_3d import Custom3DDataset
from prettytable import PrettyTable
from .risk_field_generator import HighResRiskFieldGenerator
from mmcv.datasets.nuscenes_styled_eval_utils import DetectionMetrics, EvalBoxes, DetectionBox, center_distance, accumulate, DetectionMetricDataList, calc_ap, calc_tp, quaternion_yaw

@DATASETS.register_module()
class B2D_Risk_Dataset(Custom3DDataset):
    CAN_YAW_RAD_INDEX = 16
    CAN_YAW_DEG_INDEX = 17

    
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 file_client_args=dict(backend='disk'),
                 queue_length=5,
                 bev_size=(200, 200),
                 overlap_test=False,
                 sample_interval=5,
                 name_mapping=None,
                 eval_cfg=None,
                 map_root=None,
                 map_file=None,
                 past_frames=4,
                 future_frames=4,
                 predict_frames=12,
                 planning_frames=6,
                 risk_future_frames=6,
                 point_cloud_range=[-51.2, -51.2, -2.0, 51.2, 51.2, 3.0],
                 high_res_risk_size=(512, 512),
                 risk_threshold=0.0,
                 max_influence_distance=1.0,
                 road_centerline_dilate_m=2.0,
                 road_boundary_dilate_m=0.5,
                 road_mask_closing_m=0.5,
                 randomize_history_queue=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 *args, **kwargs):
        
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode
        )
        self.sample_interval = sample_interval

        self.queue_length = queue_length
        self.bev_size = bev_size
        self.overlap_test = overlap_test
        self.eval_cfg = eval_cfg
        
        # map params
        self.map_root = map_root
        self.map_file = map_file
        
        # temporal params
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.predict_frames = predict_frames
        self.planning_frames = planning_frames
        self.risk_future_frames = risk_future_frames
        self.risk_horizon_frames = risk_future_frames + 1

        # feature params
        self.high_res_risk_size = high_res_risk_size
        self.risk_threshold = risk_threshold
        self.max_influence_distance = max_influence_distance
        self.road_centerline_dilate_m = road_centerline_dilate_m
        self.road_boundary_dilate_m = road_boundary_dilate_m
        self.road_mask_closing_m = road_mask_closing_m
        self.randomize_history_queue = randomize_history_queue

        # spatial range
        self.point_cloud_range = np.array(point_cloud_range)
        self.map_x_min = self.point_cloud_range[0]
        self.map_y_min = self.point_cloud_range[1]
        self.map_x_max = self.point_cloud_range[3]
        self.map_y_max = self.point_cloud_range[4]
        self.map_x_range = self.map_x_max - self.map_x_min
        self.map_y_range = self.map_y_max - self.map_y_min
        
        # map element class mapping
        self.map_element_class = {'Broken':0, 'Solid':1, 'SolidSolid':2,'Center':3,'TrafficLight':4,'StopSign':5}
        self.map_element_reverse_mapping = {v: k for k, v in self.map_element_class.items()}
        with open(self.map_file,'rb') as f: 
            self.map_infos = pickle.load(f)

        # name mapping
        self.NameMapping = name_mapping 
        
        # object risk weights by class id
        object_risk_weights = {}
        default_weights = {
            'car': 1.0, 'truck': 1.2, 'bus': 1.0, 'construction_vehicle': 1.2,
            'bicycle': 2.0, 'motorcycle': 2.0, 'pedestrian': 2.0, 'trailer': 1.2,
            'van': 1.2, 'speed_limit': 1.0, 'traffic_light': 1.0, 'stop': 1.0,
            'traffic_sign': 1.0, 'traffic_cone': 1.0, 'others': 1.0
        }
        for i, class_name in enumerate(self.CLASSES):
            object_risk_weights[i] = default_weights.get(class_name, 1.0)
        
        self.risk_field_generator = HighResRiskFieldGenerator(
            high_res_risk_size=self.high_res_risk_size,
            point_cloud_range=self.point_cloud_range.tolist(),
            risk_threshold=self.risk_threshold,
            max_influence_distance=self.max_influence_distance,
            road_centerline_dilate_m=self.road_centerline_dilate_m,
            road_boundary_dilate_m=self.road_boundary_dilate_m,
            road_mask_closing_m=self.road_mask_closing_m,
            object_risk_weights=object_risk_weights
        )
        

    def invert_pose(self, pose):
        inv_pose = np.eye(4)
        inv_pose[:3, :3] = np.transpose(pose[:3, :3])
        inv_pose[:3, -1] = - inv_pose[:3, :3] @ pose[:3, -1]
        return inv_pose

    def get_map_info(self, index):
        
        ann_info = self.data_infos[index]
        town_name = ann_info['town_name']
        map_info = self.map_infos[town_name]
        lane_points = map_info['lane_points']
        lane_sample_points = map_info['lane_sample_points']
        lane_types = map_info['lane_types']
        trigger_volumes_points = map_info['trigger_volumes_points']
        trigger_volumes_sample_points = map_info['trigger_volumes_sample_points']
        trigger_volumes_types = map_info['trigger_volumes_types']
        world2lidar = np.array(ann_info['sensors']['LIDAR_TOP']['world2lidar'])
        ego_xy = np.linalg.inv(world2lidar)[0:2,3]

        #1st search
        max_distance = 100
        chosed_idx = []
        for idx in range(len(lane_sample_points)):
            single_sample_points = lane_sample_points[idx]
            distance = np.linalg.norm((single_sample_points[:,0:2]-ego_xy),axis=-1)
            if np.min(distance) < max_distance:
                chosed_idx.append(idx)

        chosen_trigger_idx = []
        for idx in range(len(trigger_volumes_sample_points)):
            single_sample_point = trigger_volumes_sample_points[idx]
            distance = np.linalg.norm(single_sample_point[0:2] - ego_xy)
            if distance < max_distance:
                chosen_trigger_idx.append(idx)

        # high-res map masks
        gt_masks_high = []
        gt_labels_high = []
        gt_bboxes_high = []
        for idx in chosed_idx:
            if not lane_types[idx] in self.map_element_class.keys():
                continue
            points = lane_points[idx]
            points = np.concatenate([points,np.ones((points.shape[0],1))],axis=-1)
            points_in_ego = (world2lidar @ points.T).T
            mask = (points_in_ego[:,0]>self.point_cloud_range[0]) & (points_in_ego[:,0]<self.point_cloud_range[3]) & (points_in_ego[:,1]>self.point_cloud_range[1]) & (points_in_ego[:,1]<self.point_cloud_range[4])
            points_in_ego_range = points_in_ego[mask,0:2]
            if len(points_in_ego_range) > 1:
                gt_mask_high = np.zeros(self.high_res_risk_size,dtype=np.uint8)
                normalized_points = np.zeros_like(points_in_ego_range)
                normalized_points[:,0] = (points_in_ego_range[:,0] - self.map_x_min) * (self.high_res_risk_size[0] / self.map_x_range)
                normalized_points[:,1] = (points_in_ego_range[:,1] - self.map_y_min) * (self.high_res_risk_size[1] / self.map_y_range)
                cv2.polylines(gt_mask_high, [normalized_points.astype(np.int32)], False, color=1, thickness=2)
                gt_label_high =  self.map_element_class[lane_types[idx]]
                gt_masks_high.append(gt_mask_high)
                gt_labels_high.append(gt_label_high)
                ys, xs = np.where(gt_mask_high==1)
                gt_bboxes_high.append([min(xs), min(ys), max(xs), max(ys)]) 

        for idx in chosen_trigger_idx:
            if trigger_volumes_types[idx] not in self.map_element_class:
                continue
            points = trigger_volumes_points[idx]
            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=-1)
            points_in_ego = (world2lidar @ points.T).T
            mask = (
                (points_in_ego[:, 0] > self.point_cloud_range[0]) &
                (points_in_ego[:, 0] < self.point_cloud_range[3]) &
                (points_in_ego[:, 1] > self.point_cloud_range[1]) &
                (points_in_ego[:, 1] < self.point_cloud_range[4])
            )
            points_in_ego_range = points_in_ego[mask, 0:2]
            if len(points_in_ego_range) < 3:
                continue
            gt_mask_high = np.zeros(self.high_res_risk_size, dtype=np.uint8)
            normalized_points = np.zeros_like(points_in_ego_range)
            normalized_points[:, 0] = (
                (points_in_ego_range[:, 0] - self.map_x_min) *
                (self.high_res_risk_size[0] / self.map_x_range)
            )
            normalized_points[:, 1] = (
                (points_in_ego_range[:, 1] - self.map_y_min) *
                (self.high_res_risk_size[1] / self.map_y_range)
            )
            cv2.fillConvexPoly(gt_mask_high, normalized_points.astype(np.int32), color=1)
            gt_label_high = self.map_element_class[trigger_volumes_types[idx]]
            gt_masks_high.append(gt_mask_high)
            gt_labels_high.append(gt_label_high)
            ys, xs = np.where(gt_mask_high == 1)
            gt_bboxes_high.append([min(xs), min(ys), max(xs), max(ys)])

        if len(gt_masks_high) == 0:
            gt_masks_high.append(np.zeros(self.high_res_risk_size,dtype=np.uint8))
            gt_labels_high.append(-1)
            gt_bboxes_high.append([0,0,0,0])

        gt_masks_high = np.stack(gt_masks_high)
        gt_labels_high = np.array(gt_labels_high)
        gt_bboxes_high = np.array(gt_bboxes_high)

        return gt_masks_high,gt_labels_high,gt_bboxes_high


    def get_data_info(self, index, is_current_frame=True):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.
            is_current_frame (bool): whether this is the current frame for supervision.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines.
        """
        info = self.data_infos[index]

        # apply name mapping
        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]

        # basic fields
        world2lidar = np.asarray(info['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float32)
        ego_yaw = np.nan_to_num(info['ego_yaw'], nan=np.pi / 2)
        ego2global = np.eye(4, dtype=np.float32)
        ego2global[:3, :3] = Quaternion(axis=[0, 0, 1], radians=ego_yaw).rotation_matrix
        ego2global[:3, 3] = np.asarray(info['ego_translation'], dtype=np.float32)
        input_dict = dict(
            folder=info['folder'],
            scene_token=info['folder'],
            sample_idx=info['folder'] + '_' + str(info['frame_idx']),
            frame_idx=info['frame_idx'],
            ego_yaw=ego_yaw,
            ego_translation=info['ego_translation'],
            sensors=info['sensors'],
            world2lidar=world2lidar,
            ego2global=ego2global,
            world2ego=world2lidar,
            gt_ids=info['gt_ids'],
            gt_boxes=info['gt_boxes'], # right hand
            gt_names=info['gt_names'],
            ego_vel=info['ego_vel'],
            ego_accel=info['ego_accel'],
            ego_rotation_rate=info['ego_rotation_rate'],
            npc2world=info['npc2world'],
        )
        # image data
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            lidar2ego = info['sensors']['LIDAR_TOP']['lidar2ego']
            for sensor_type, cam_info in info['sensors'].items():
                if not 'CAM' in sensor_type:
                    continue
                image_paths.append(osp.join(self.data_root,cam_info['data_path']))
                # obtain lidar to image transformation matrix
                cam2ego = cam_info['cam2ego']
                intrinsic = cam_info['intrinsic']
                intrinsic_pad = np.eye(4)
                intrinsic_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2cam = self.invert_pose(cam2ego) @ lidar2ego
                lidar2img = intrinsic_pad @ lidar2cam
                lidar2img_rts.append(lidar2img)
                cam_intrinsics.append(intrinsic_pad)
                lidar2cam_rts.append(lidar2cam)
            lidar2global = ego2global @ lidar2ego
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    lidar2global=lidar2global,
                ))

        # annotation
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        # add all annotation fields to input_dict
        for key, value in annos.items():
            input_dict[key] = value
        yaw = input_dict['ego_yaw']
        rotation = list(Quaternion(axis=[0, 0, 1], radians=yaw))
        if yaw < 0:
            yaw += 2*np.pi
        yaw_in_degree = yaw / np.pi * 180 
        
        can_bus = np.zeros(18)
        can_bus[:3] = input_dict['ego_translation']
        can_bus[3:7] = rotation
        can_bus[7:10] = input_dict['ego_vel']
        can_bus[10:13] = input_dict['ego_accel']
        can_bus[13:16] = input_dict['ego_rotation_rate']
        can_bus[16] = yaw
        can_bus[17] = yaw_in_degree
        input_dict['can_bus'] = can_bus
        if is_current_frame:
            ego_lcf_feat = np.zeros(9, dtype=np.float32)
            ego_lcf_feat[0:2] = np.asarray(input_dict['ego_translation'][0:2], dtype=np.float32)
            ego_lcf_feat[2:4] = np.asarray(input_dict['ego_accel'][0:2], dtype=np.float32)
            ego_lcf_feat[4] = float(input_dict['ego_rotation_rate'][-1])
            ego_lcf_feat[5] = float(info['ego_size'][1])
            ego_lcf_feat[6] = float(info['ego_size'][0])
            ego_lcf_feat[7] = float(np.linalg.norm(np.asarray(input_dict['ego_vel'][0:2], dtype=np.float32)))
            ego_lcf_feat[8] = float(info.get('steer', 0.0))
            input_dict['ego_lcf_feat'] = ego_lcf_feat
            ego_his_trajs, ego_his_trajs_mask = self.get_ego_history_traj(index, self.sample_interval, self.past_frames)
            input_dict['ego_his_trajs'] = ego_his_trajs
            input_dict['ego_his_trajs_mask'] = ego_his_trajs_mask
            gt_masks_high, gt_labels_high, gt_bboxes_high = self.get_map_info(index)
            sdc_planning, sdc_planning_mask = self.get_ego_future_xy(index, self.sample_interval, self.planning_frames)
            input_dict['sdc_planning'] = sdc_planning
            input_dict['sdc_planning_mask'] = sdc_planning_mask
            command_val = info.get('command_near', 4)
            if command_val <= 0 or command_val > 6:
                command_val = 4
            command_val -= 1
            input_dict['command'] = command_val
            far_xy_world = np.asarray(info['command_far_xy'], dtype=np.float32)
            goal_x = world2lidar[0, 0] * far_xy_world[0] + world2lidar[0, 1] * far_xy_world[1] + world2lidar[0, 3]
            goal_y = world2lidar[1, 0] * far_xy_world[0] + world2lidar[1, 1] * far_xy_world[1] + world2lidar[1, 3]
            input_dict['goal_point'] = np.array([goal_x, goal_y], dtype=np.float32)
            gt_risk_score, gt_risk_valid_mask = self._generate_temporal_risk_field(index)
            input_dict['gt_risk_score'] = gt_risk_score
            input_dict['gt_risk_valid_mask'] = gt_risk_valid_mask
        else:
            input_dict['ego_lcf_feat'] = np.zeros(9, dtype=np.float32)
            input_dict['ego_his_trajs'] = np.zeros((self.past_frames, 2), dtype=np.float32)
            input_dict['ego_his_trajs_mask'] = np.zeros((self.past_frames, 2), dtype=np.float32)
            input_dict['sdc_planning'] = np.zeros((1, self.planning_frames, 3), dtype=np.float32)
            input_dict['sdc_planning_mask'] = np.zeros((1, self.planning_frames, 2), dtype=np.float32)
            input_dict['command'] = 0
            input_dict['goal_point'] = np.zeros(2, dtype=np.float32)
            input_dict['gt_risk_score'] = np.zeros((1, 1, 1, 1), dtype=np.float32)
            input_dict['gt_risk_valid_mask'] = np.zeros((1, 1, 1, 1), dtype=np.float32)
        
        return input_dict

    def _normalize_angle(self, angle):
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _map_names_and_labels(self, names):
        mapped_names = []
        labels = []
        for name in names:
            mapped_name = self.NameMapping.get(name, name)
            mapped_names.append(mapped_name)
            labels.append(self.CLASSES.index(mapped_name) if mapped_name in self.CLASSES else -1)
        return np.array(mapped_names), np.array(labels)

    def _boxes_in_reference_lidar(self, ref_index, target_index):
        ref_info = self.data_infos[ref_index]
        target_info = self.data_infos[target_index]
        world2lidar_ref = np.asarray(ref_info['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float32)
        lidar2world_target = np.linalg.inv(np.asarray(target_info['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float32))
        target_boxes = np.asarray(target_info['gt_boxes'], dtype=np.float32)
        if target_boxes.size == 0:
            return target_boxes.reshape(0, 9)
        boxes = target_boxes.copy()
        target_to_ref_rot = world2lidar_ref[:3, :3] @ lidar2world_target[:3, :3]
        for i in range(target_boxes.shape[0]):
            center_target = np.concatenate([target_boxes[i, 0:3], np.ones(1, dtype=np.float32)])
            boxes[i, 0:3] = (world2lidar_ref @ lidar2world_target @ center_target)[:3]
            yaw_target = -(target_boxes[i, 6] + np.pi / 2)
            heading_target = np.array([np.cos(yaw_target), np.sin(yaw_target), 0.0], dtype=np.float32)
            heading_ref = target_to_ref_rot @ heading_target
            yaw_ref = np.arctan2(heading_ref[1], heading_ref[0])
            boxes[i, 6] = -self._normalize_angle(yaw_ref) - np.pi / 2
            if target_boxes.shape[1] > 8:
                velocity_ref = target_to_ref_rot @ np.array([target_boxes[i, 7], target_boxes[i, 8], 0.0], dtype=np.float32)
                boxes[i, 7:9] = velocity_ref[:2]
        return boxes

    def _generate_single_risk_components(self, ref_index, target_index):
        ref_info = self.data_infos[ref_index]
        target_info = self.data_infos[target_index]
        if target_info['folder'] != ref_info['folder']:
            return None
        gt_masks_high, gt_labels_high, gt_bboxes_high = self.get_map_info(ref_index)
        gt_names, gt_labels = self._map_names_and_labels(target_info['gt_names'])
        components = self.risk_field_generator.generate_risk_components(
            gt_boxes=self._boxes_in_reference_lidar(ref_index, target_index),
            gt_names=gt_names,
            gt_labels=gt_labels,
            ego_vel=ref_info['ego_vel'][0],
            weather_info=target_info['weather'],
            gt_masks_high=gt_masks_high,
            gt_labels_high=gt_labels_high,
            gt_bboxes_high=gt_bboxes_high,
            traffic_elements_info=target_info['traffic_elements_info'],
            world2lidar=np.asarray(ref_info['sensors']['LIDAR_TOP']['world2lidar'], dtype=np.float32),
        )
        risk_score = components['final'][np.newaxis, :, :]
        valid_mask = components['road_mask'].astype(np.float32)[np.newaxis, :, :]
        return risk_score.astype(np.float32), valid_mask

    def _generate_temporal_risk_field(self, index):
        risk_score = np.zeros(
            (self.risk_horizon_frames, 1, self.high_res_risk_size[0], self.high_res_risk_size[1]),
            dtype=np.float32)
        valid_mask = np.zeros(
            (self.risk_horizon_frames, 1, self.high_res_risk_size[0], self.high_res_risk_size[1]),
            dtype=np.float32)
        for horizon_idx in range(self.risk_horizon_frames):
            target_index = index + horizon_idx * self.sample_interval
            if target_index >= len(self.data_infos):
                break
            components = self._generate_single_risk_components(index, target_index)
            if components is None:
                break
            risk_score[horizon_idx], valid_mask[horizon_idx] = components
        return np.clip(risk_score, 0.0, 1.0), valid_mask


    def prepare_train_data(self, index):
        
        queue = []

        # build history queue with sample_interval
        index_list = list(range(index-self.queue_length*self.sample_interval, index, self.sample_interval))
        if self.randomize_history_queue:
            random.shuffle(index_list)
            index_list = sorted(index_list[1:])
        else:
            index_list = index_list[1:]
        index_list.append(index)
        
        for i in index_list:
            i = max(0, i)
            input_dict = self.get_data_info(i, is_current_frame=(i == index))
            if input_dict is None:
                return None
            
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            queue.append(example)
        
        # merge temporal data
        merged_data = self.union2one(queue)
        return merged_data
    

    def get_ann_info(self, index):
        
        # get basic annotation
        info = self.data_infos[index]
        
        # apply name mapping
        for i in range(len(info['gt_names'])):
            if info['gt_names'][i] in self.NameMapping.keys():
                info['gt_names'][i] = self.NameMapping[info['gt_names'][i]]
        # filter valid boxes
        mask = (info['num_points'] >= -1)
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)  # unknown class
        gt_labels_3d = np.array(gt_labels_3d)
        # create LiDAR boxes
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
        # build annotation dict
        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            )
        return anns_results


    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index, is_current_frame=True)
        self.pre_pipeline(input_dict)
        return self.pipeline(input_dict)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)

        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data


    def union2one(self, queue):
        imgs_list = [each['img'].data for each in queue]

        metas_map = {}
        prev_scene_token = None
        prev_world2lidar = None
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            cur_world2lidar = np.asarray(metas_map[i]['world2lidar'], dtype=np.float32)
            if metas_map[i]['folder'] != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['folder']
                metas_map[i]['can_bus'][:3] = 0
                metas_map[i]['can_bus'][self.CAN_YAW_RAD_INDEX] = 0
                metas_map[i]['can_bus'][self.CAN_YAW_DEG_INDEX] = 0
            else:
                metas_map[i]['prev_bev_exists'] = True
                prev_from_cur = prev_world2lidar @ np.linalg.inv(cur_world2lidar)
                delta_yaw = self._normalize_angle(np.arctan2(prev_from_cur[1, 0], prev_from_cur[0, 0]))
                metas_map[i]['can_bus'][:3] = prev_from_cur[:3, 3]
                metas_map[i]['can_bus'][self.CAN_YAW_RAD_INDEX] = delta_yaw
                metas_map[i]['can_bus'][self.CAN_YAW_DEG_INDEX] = delta_yaw / np.pi * 180
            prev_world2lidar = cur_world2lidar

        gt_risk_score = queue[-1]['gt_risk_score'].data
        gt_risk_valid_mask = queue[-1]['gt_risk_valid_mask'].data

        # planning fields (current frame only)
        # fields already wrapped in DataContainer
        sdc_planning = queue[-1]['sdc_planning'].data
        sdc_planning_mask = queue[-1]['sdc_planning_mask'].data
        command = queue[-1]['command'].data
        ego_his_trajs = queue[-1]['ego_his_trajs'].data
        ego_his_trajs_mask = queue[-1]['ego_his_trajs_mask'].data
        ego_lcf_feat = queue[-1]['ego_lcf_feat'].data
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        img_metas_list = [metas_map[i] for i in sorted(metas_map)]
        queue[-1]['img_metas'] = DC(img_metas_list, cpu_only=True)
        queue = queue[-1]
        if gt_risk_score is not None:
            queue['gt_risk_score'] = DC(gt_risk_score)
        if gt_risk_valid_mask is not None:
            queue['gt_risk_valid_mask'] = DC(gt_risk_valid_mask)
        # stack ego history trajectories
        if 'gt_sdc_past_traj' in queue:
            sdc_past_list = [torch.as_tensor(each['gt_sdc_past_traj'].data if hasattr(each['gt_sdc_past_traj'], 'data') else each['gt_sdc_past_traj']) for each in queue]
            queue['gt_sdc_past_traj'] = DC(torch.stack(sdc_past_list), cpu_only=False, stack=True)
        # add planning fields to output
        queue['sdc_planning'] = DC(sdc_planning)
        queue['sdc_planning_mask'] = DC(sdc_planning_mask)
        queue['command'] = DC(command)
        queue['ego_his_trajs'] = DC(ego_his_trajs)
        queue['ego_his_trajs_mask'] = DC(ego_his_trajs_mask)
        queue['ego_lcf_feat'] = DC(ego_lcf_feat)
        return queue

    def get_past_or_future_xy(self,idx,sample_rate,frames,past_or_future,local_xy=False):
        assert past_or_future in ['past','future']
        if past_or_future == 'past':
            adj_idx_list = range(idx-sample_rate,idx-(frames+1)*sample_rate,-sample_rate)
        else:
            adj_idx_list = range(idx+sample_rate,idx+(frames+1)*sample_rate,sample_rate)

        cur_frame = self.data_infos[idx]
        box_ids = cur_frame['gt_ids']
        adj_track = np.zeros((len(box_ids),frames,2))
        adj_mask = np.zeros((len(box_ids),frames,2))
        world2lidar_ego_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for i in range(len(box_ids)):
            box_id = box_ids[i]
            cur_box2lidar = world2lidar_ego_cur @ cur_frame['npc2world'][i]
            cur_xy = cur_box2lidar[0:2,3]
            for j in range(len(adj_idx_list)):
                adj_idx = adj_idx_list[j]
                if adj_idx <0 or adj_idx>=len(self.data_infos):
                    break
                adj_frame = self.data_infos[adj_idx]
                if adj_frame['folder'] != cur_frame ['folder']:
                    break
                ids_where = np.where(adj_frame['gt_ids']==box_id)[0]
                if len(ids_where) != 1:
                    continue
                adj_ind = ids_where[0]
                adj_box2lidar = world2lidar_ego_cur @ adj_frame['npc2world'][adj_ind]
                adj_xy = adj_box2lidar[0:2,3]
                if local_xy:
                    adj_xy -= cur_xy
                adj_track[i,j,:] = adj_xy
                adj_mask[i,j,:] = 1
        return adj_track, adj_mask

    def get_ego_history_traj(self, idx, sample_rate, frames):
        adj_idx_list = range(idx-frames*sample_rate, idx+sample_rate, sample_rate)
        cur_frame = self.data_infos[idx]
        full_track = np.zeros((frames + 1, 2), dtype=np.float32)
        full_mask = np.zeros(frames + 1, dtype=np.float32)
        world2lidar_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j, adj_idx in enumerate(adj_idx_list):
            if adj_idx < 0 or adj_idx >= len(self.data_infos):
                continue
            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame['folder']:
                continue
            world2lidar_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_cur @ np.linalg.inv(world2lidar_adj)
            full_track[j] = adj2cur_lidar[0:2, 3]
            full_mask[j] = 1.0
        offset_track = full_track[1:] - full_track[:-1]
        offset_mask = np.minimum(full_mask[1:], full_mask[:-1])
        for j in range(frames - 1, -1, -1):
            if offset_mask[j] == 0 and j + 1 < frames:
                offset_track[j] = offset_track[j + 1]
        return offset_track.astype(np.float32), offset_mask[:, None].repeat(2, axis=1).astype(np.float32)

    def get_ego_future_xy(self,idx,sample_rate,frames):
        adj_idx_list = range(idx+sample_rate,idx+(frames+1)*sample_rate,sample_rate)
        cur_frame = self.data_infos[idx]
        adj_track = np.zeros((1,frames,3))
        adj_mask = np.zeros((1,frames,2))
        world2lidar_ego_cur = cur_frame['sensors']['LIDAR_TOP']['world2lidar']
        for j in range(len(adj_idx_list)):
            adj_idx = adj_idx_list[j]
            if adj_idx <0 or adj_idx>=len(self.data_infos):
                break
            adj_frame = self.data_infos[adj_idx]
            if adj_frame['folder'] != cur_frame ['folder']:
                break
            world2lidar_ego_adj = adj_frame['sensors']['LIDAR_TOP']['world2lidar']
            adj2cur_lidar = world2lidar_ego_cur @ np.linalg.inv(world2lidar_ego_adj)
            xy = adj2cur_lidar[0:2,3]
            yaw = np.arctan2(adj2cur_lidar[1,0],adj2cur_lidar[0,0])
            yaw = -yaw -np.pi
            while yaw > np.pi:
                yaw -= np.pi*2
            while yaw < -np.pi:
                yaw += np.pi*2
            adj_track[0,j,0:2] = xy
            adj_track[0,j,2] = yaw
            adj_mask[0,j,:] = 1
        return adj_track, adj_mask


    def evaluate(self,
                 results,
                 metric=('risk', 'planning'),
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluate risk and planning outputs, with bbox kept as optional."""
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = list(metric)

        detail = dict()
        tmp_dir = None
        has_bbox = False
        bbox_results = results
        if isinstance(results, dict) and 'bbox_results' in results:
            bbox_results = results['bbox_results']
            has_bbox = True
        elif isinstance(results, list) and len(results) > 0:
            has_bbox = 'pts_bbox' in results[0] or 'img_bbox' in results[0] or 'boxes_3d' in results[0]

        if 'bbox' in metrics and has_bbox:
            result_files, tmp_dir = self.format_results(bbox_results, jsonfile_prefix)
            if isinstance(result_files, dict):
                result_path = next(iter(result_files.values()))
            else:
                result_path = result_files

            with open(result_path) as f:
                result_data = json.load(f)
            pred_boxes = EvalBoxes.deserialize(result_data['results'], DetectionBox)
            meta = result_data['meta']
            gt_boxes = self.load_gt()

            metric_data_list = DetectionMetricDataList()
            for class_name in self.eval_cfg['class_names']:
                for dist_th in self.eval_cfg['dist_ths']:
                    md = accumulate(gt_boxes, pred_boxes, class_name, center_distance, dist_th)
                    metric_data_list.set(class_name, dist_th, md)

            metrics_summary = DetectionMetrics(self.eval_cfg)
            for class_name in self.eval_cfg['class_names']:
                for dist_th in self.eval_cfg['dist_ths']:
                    metric_data = metric_data_list[(class_name, dist_th)]
                    ap = calc_ap(metric_data, self.eval_cfg['min_recall'], self.eval_cfg['min_precision'])
                    metrics_summary.add_label_ap(class_name, dist_th, ap)
                for metric_name in self.eval_cfg['tp_metrics']:
                    metric_data = metric_data_list[(class_name, self.eval_cfg['dist_th_tp'])]
                    tp = calc_tp(metric_data, self.eval_cfg['min_recall'], metric_name)
                    metrics_summary.add_label_tp(class_name, metric_name, tp)
            metrics_summary = metrics_summary.serialize()
            metrics_summary['meta'] = meta.copy()
            detail['bbox_NuScenes/NDS'] = metrics_summary['nd_score']
            detail['bbox_NuScenes/mAP'] = metrics_summary['mean_ap']

        if 'risk' in metrics:
            detail.update(self._evaluate_risk_field(results))
        if 'planning' in metrics:
            detail.update(self._evaluate_planning(results))

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return detail

    def _evaluate_risk_field(self, results):
        """Evaluate risk field predictions."""
        abs_error_sum = 0.0
        sq_error_sum = 0.0
        valid_count = 0.0

        if isinstance(results, dict):
            iterable = results.get('risk_results', results.get('bbox_results', []))
        else:
            iterable = results
        for result in iterable:
            if not isinstance(result, dict):
                continue
            if 'risk_abs_error_sum' not in result or 'risk_sq_error_sum' not in result or 'risk_valid_count' not in result:
                continue
            abs_error_sum += float(result['risk_abs_error_sum'])
            sq_error_sum += float(result['risk_sq_error_sum'])
            valid_count += float(result['risk_valid_count'])

        eval_results = {}
        if valid_count > 0:
            eval_results['risk_mae'] = abs_error_sum / valid_count
            eval_results['risk_rmse'] = float(np.sqrt(sq_error_sum / valid_count))
        return eval_results

    def _evaluate_planning(self, results):
        """Evaluate planning trajectory results."""
        pred_trajs = []
        gt_trajs = []
        gt_masks = []

        if isinstance(results, dict):
            iterable = results.get('planning_results', results.get('bbox_results', []))
        else:
            iterable = results
        for result in iterable:
            if not isinstance(result, dict):
                continue
            if 'planning_traj' not in result or 'planning_traj_gt' not in result:
                continue
            pred_trajs.append(np.asarray(result['planning_traj'], dtype=np.float32))
            gt_trajs.append(np.asarray(result['planning_traj_gt'], dtype=np.float32))
            mask = result.get('planning_traj_mask', None)
            if mask is None:
                gt_masks.append(np.ones_like(gt_trajs[-1][..., :2], dtype=np.float32))
            else:
                gt_masks.append(np.asarray(mask, dtype=np.float32))

        eval_results = {}
        if pred_trajs and gt_trajs:
            ade_values = []
            fde_values = []
            for pred, gt, mask in zip(pred_trajs, gt_trajs, gt_masks):
                valid = mask[..., 0] > 0
                if not np.any(valid):
                    continue
                step_l2 = np.linalg.norm(pred[..., :2] - gt[..., :2], axis=-1)
                ade_values.append(float(step_l2[valid].mean()))
                last_valid = np.where(valid)[0][-1]
                fde_values.append(float(step_l2[last_valid]))
            if ade_values:
                eval_results['planning_ADE'] = float(np.mean(ade_values))
                eval_results['planning_FDE'] = float(np.mean(fde_values))
        return eval_results

    def load_gt(self):
        all_annotations = EvalBoxes()
        for i in range(len(self.data_infos)):
            sample_boxes = []
            sample_data = self.data_infos[i]
            gt_boxes = sample_data.get('gt_boxes', None)
            if gt_boxes is None:
                continue
            for j in range(gt_boxes.shape[0]):
                if j >= len(sample_data['gt_names']):
                    continue
                class_name = self.NameMapping.get(sample_data['gt_names'][j], sample_data['gt_names'][j])
                if not class_name in self.eval_cfg['class_range'].keys():
                    continue
                range_x, range_y = self.eval_cfg['class_range'][class_name]
                if abs(gt_boxes[j,0]) > range_x or abs(gt_boxes[j,1]) > range_y:
                    continue
                num_points_arr = sample_data.get('num_points', None)
                if isinstance(num_points_arr, (list, np.ndarray)):
                    num_pts = int(num_points_arr[j]) if j < len(num_points_arr) else 0
                else:
                    num_pts = 0
                sample_boxes.append(DetectionBox(
                    sample_token=sample_data['folder']+'_'+str(sample_data['frame_idx']),
                    translation=gt_boxes[j,0:3],
                    size=gt_boxes[j,3:6],
                    rotation=list(Quaternion(axis=[0, 0, 1], radians=-gt_boxes[j,6]-np.pi/2)),
                    velocity=gt_boxes[j,7:9] if gt_boxes.shape[1] > 7 else [0.0, 0.0],
                    num_pts=num_pts,
                    detection_name=class_name,
                    detection_score=-1.0,
                    attribute_name=class_name
                ))
            all_annotations.add_boxes(sample_data['folder']+'_'+str(sample_data['frame_idx']), sample_boxes)
        return all_annotations

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert detection results to NuScenes format."""

        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(track_iter_progress(results)):
            if 'boxes_3d' not in det:
                continue
                
            annos = []
            box3d = det['boxes_3d']
            scores = det['scores_3d']
            labels = det['labels_3d']
            
            if len(box3d) == 0:
                continue
                
            box_gravity_center = box3d.gravity_center
            box_dims = box3d.dims
            box_yaw = box3d.yaw.numpy()
            box_yaw = -box_yaw - np.pi / 2
            
            sample_token = self.data_infos[sample_id]['folder'] + '_' + str(self.data_infos[sample_id]['frame_idx'])

            for i in range(len(box3d)):
                quat = list(Quaternion(axis=[0, 0, 1], radians=box_yaw[i]))
                velocity = [box3d.tensor[i, 7].item(), box3d.tensor[i, 8].item()] if box3d.tensor.shape[1] > 7 else [0.0, 0.0]
                name = mapped_class_names[labels[i]]
                
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box_gravity_center[i].tolist(),
                    size=box_dims[i].tolist(),
                    rotation=quat,
                    velocity=velocity,
                    detection_name=name,
                    detection_score=scores[i].item(),
                    attribute_name=name
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
            
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format results to JSON."""
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir
