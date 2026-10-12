#!/usr/bin/env python3

import numpy as np
import cv2
from typing import Dict, List, Any, Tuple

# map risk weights for the 6 B2D map elements
MAP_RISK_WEIGHTS = {
    'Broken': 0.3,
    'Solid': 0.6,
    'SolidSolid': 0.8,
    'Center': 0.0,
    'TrafficLight': 0.3,
    'StopSign': 0.6
}

class HighResRiskFieldGenerator:
    
    def __init__(self,
                 high_res_risk_size: Tuple[int, int] = (512, 512),
                 point_cloud_range: List[float] = [-51.2, -51.2, -2.0, 51.2, 51.2, 3.0],
                 risk_threshold: float = 0.0,
                 max_influence_distance: float = 0.8,
                 exponent_clip_max: float = 100.0,  # clip exponent to prevent exp() overflow
                 min_sigma: float = 0.1,  # min sigma to prevent division by zero
                 # road mask params
                 road_centerline_dilate_m: float = 2.0,
                 road_boundary_dilate_m: float = 0.5,
                 road_mask_closing_m: float = 0.5,
                 # object risk weights
                 object_risk_weights: Dict[int, float] = None):  # object risk weights
        self.high_res_risk_size = high_res_risk_size
        self.point_cloud_range = point_cloud_range
        self.risk_threshold = risk_threshold
        
        # configurable params
        self.max_influence_distance = max_influence_distance
        
        # numerical stability params
        self.exponent_clip_max = exponent_clip_max
        self.min_sigma = min_sigma
        
        # road mask params
        self.road_centerline_dilate_m = road_centerline_dilate_m
        self.road_boundary_dilate_m = road_boundary_dilate_m
        self.road_mask_closing_m = road_mask_closing_m
        
        # object risk weights
        self.object_risk_weights = object_risk_weights
        
        # physical coordinate range
        self.x_min, self.y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        self.x_max, self.y_max = self.point_cloud_range[3], self.point_cloud_range[4]
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        
        # high-res size
        # size is parsed as (Width, Height): Col=X (forward), Row=Y (lateral)
        self.w, self.h = self.high_res_risk_size
        
        # pixel resolution
        self.pixels_per_meter_x, self.pixels_per_meter_y = self._get_pixels_per_meter()

        # ego pixel coords (col=x, row=y)
        # Ego is at (0, 0) in physical coordinates
        self.ego_col = int((0 - self.x_min) * self.w / self.x_range)
        self.ego_row = int((0 - self.y_min) * self.h / self.y_range)
        
        # whether ego is in bounds
        self.ego_in_bounds = (0 <= self.ego_col < self.w) and (0 <= self.ego_row < self.h)
    
    
    def _get_pixels_per_meter(self):
        # x_range maps to width (col)
        pixels_per_meter_x = self.w / self.x_range
        # y_range maps to height (row)
        pixels_per_meter_y = self.h / self.y_range
        return pixels_per_meter_x, pixels_per_meter_y

    def _build_ellipse_kernel(self, radius_m: float) -> np.ndarray:
        radius_x = int(round(radius_m * self.pixels_per_meter_x))
        radius_y = int(round(radius_m * self.pixels_per_meter_y))
        return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius_x * 2 + 1, radius_y * 2 + 1),
        )

    def _dilate_mask(self, mask: np.ndarray, radius_m: float) -> np.ndarray:
        if radius_m <= 0.0:
            return mask.astype(bool)
        kernel = self._build_ellipse_kernel(radius_m)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    def _close_mask(self, mask: np.ndarray, radius_m: float) -> np.ndarray:
        if radius_m <= 0.0:
            return mask.astype(bool)
        kernel = self._build_ellipse_kernel(radius_m)
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

    def _merge_union(self, base: np.ndarray, new_risk: np.ndarray) -> np.ndarray:
        new_risk = np.clip(new_risk.astype(np.float32), 0.0, 1.0)
        return 1.0 - (1.0 - base) * (1.0 - new_risk)

    def _create_road_boundary_mask(self,
                                   gt_masks_high: np.ndarray,
                                   gt_labels_high: np.ndarray,
                                   gt_bboxes_high: np.ndarray) -> np.ndarray:
        h, w = self.high_res_risk_size
        
        # 1. separate map element types
        center_lines = gt_masks_high[gt_labels_high == 3]  # lane centerline
        other_lines = gt_masks_high[np.isin(gt_labels_high, [0, 1, 2])]  # other lane lines (Broken, Solid, SolidSolid)
        trigger_regions = gt_masks_high[np.isin(gt_labels_high, [4, 5])]  # trigger regions (TrafficLight, StopSign)
        
        # 2. merge same-type masks
        center_mask = np.any(center_lines, axis=0) if center_lines.size > 0 else np.zeros((h, w), dtype=bool)
        other_mask = np.any(other_lines, axis=0) if other_lines.size > 0 else np.zeros((h, w), dtype=bool)
        trigger_mask = np.any(trigger_regions, axis=0) if trigger_regions.size > 0 else np.zeros((h, w), dtype=bool)

        # 3. dilate to approximate drivable area
        center_dilated = self._dilate_mask(center_mask, self.road_centerline_dilate_m)
        other_dilated = self._dilate_mask(other_mask, self.road_boundary_dilate_m)

        # 4. merge and close to reduce missed road
        road_mask = center_dilated | other_dilated | trigger_mask
        road_mask = self._close_mask(road_mask, self.road_mask_closing_m)
        return road_mask

    def _build_trigger_vertices_local(self,
                                      trigger_loc: np.ndarray,
                                      trigger_size: np.ndarray,
                                      trigger_rot: np.ndarray,
                                      world2lidar: np.ndarray) -> np.ndarray:
        half_length = trigger_size[0] * 0.5
        half_width = trigger_size[1] * 0.5
        yaw = np.deg2rad(trigger_rot[2])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        vertices_local = np.array([
            [half_length, half_width],
            [half_length, -half_width],
            [-half_length, -half_width],
            [-half_length, half_width],
        ], dtype=np.float32)
        rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
        vertices_world_xy = vertices_local @ rotation.T + trigger_loc[None, :2]
        vertices_world = np.concatenate(
            [
                vertices_world_xy,
                np.full((4, 1), trigger_loc[2], dtype=np.float32),
                np.ones((4, 1), dtype=np.float32),
            ],
            axis=1,
        )
        vertices_lidar = (world2lidar @ vertices_world.T).T[:, :2]
        return vertices_lidar

    def _rasterize_polygon_mask(self, vertices_lidar: np.ndarray) -> np.ndarray:
        polygon = np.zeros((self.h, self.w), dtype=np.uint8)
        vertices_pixels = np.empty((vertices_lidar.shape[0], 2), dtype=np.float32)
        vertices_pixels[:, 0] = (vertices_lidar[:, 0] - self.x_min) * self.w / self.x_range
        vertices_pixels[:, 1] = (vertices_lidar[:, 1] - self.y_min) * self.h / self.y_range
        cv2.fillPoly(polygon, [vertices_pixels.astype(np.int32)], 1)
        return polygon.astype(bool)

    def generate_lane_risk_field(self,
                                 gt_masks_high: np.ndarray,
                                 gt_labels_high: np.ndarray,
                                 gt_bboxes_high: np.ndarray) -> np.ndarray:
        lane_risk_field = np.zeros(self.high_res_risk_size, dtype=np.float32)

        # only process the 3 lane-line types
        lane_config = {
            0: MAP_RISK_WEIGHTS['Broken'],      # Broken
            1: MAP_RISK_WEIGHTS['Solid'],        # Solid
            2: MAP_RISK_WEIGHTS['SolidSolid']    # SolidSolid
        }

        for label, base_weight in lane_config.items():
            class_mask_indices = gt_labels_high == label
            if not np.any(class_mask_indices):
                continue

            class_lane_mask = np.any(gt_masks_high[class_mask_indices], axis=0)
            if not np.any(class_lane_mask):
                continue

            in_mask = ~class_lane_mask
            if not np.any(in_mask):
                lane_risk = np.zeros((self.h, self.w), dtype=np.float32)
                lane_risk[class_lane_mask] = base_weight
                lane_risk_field = self._merge_union(lane_risk_field, lane_risk)
                continue

            src = (~class_lane_mask).astype(np.uint8) * 255
            distance_field = cv2.distanceTransform(src, distanceType=cv2.DIST_L2, maskSize=5)

            sigma_meter_y = self.max_influence_distance
            sigma_y = sigma_meter_y * self.pixels_per_meter_y

            # Gaussian decay, truncated beyond 4σ
            risk_decay = np.exp(-0.5 * (distance_field / sigma_y) ** 2)
            risk_decay[distance_field > 4 * sigma_y] = 0.0

            lane_risk = np.zeros_like(distance_field, dtype=np.float32)
            lane_risk[in_mask] = (base_weight * risk_decay[in_mask]).astype(np.float32)
            lane_risk[class_lane_mask] = base_weight

            lane_risk_field = self._merge_union(lane_risk_field, lane_risk)

        return lane_risk_field

    def generate_road_boundary_risk_field(self,
                                          road_mask: np.ndarray) -> np.ndarray:
        # morphological gradient: mask - erode(mask)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(road_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        boundary = road_mask & ~eroded
        if not np.any(boundary):
            return np.zeros(self.high_res_risk_size, dtype=np.float32)

        src = (~boundary).astype(np.uint8) * 255
        distance_field = cv2.distanceTransform(src, distanceType=cv2.DIST_L2, maskSize=5)

        sigma_meter = self.max_influence_distance
        sigma_pixel = sigma_meter * self.pixels_per_meter_y

        weight = 0.6
        risk_decay = np.exp(-0.5 * (distance_field / sigma_pixel) ** 2)
        risk_decay[distance_field > 4 * sigma_pixel] = 0.0

        risk = np.zeros_like(distance_field, dtype=np.float32)
        risk[boundary] = weight
        risk[~boundary] = (weight * risk_decay[~boundary]).astype(np.float32)
        return risk

    def generate_traffic_sign_risk_field(self,
                                         traffic_elements_info,
                                         ego_vel,
                                         world2lidar: np.ndarray) -> np.ndarray:
        
        traffic_risk_field = np.zeros((self.h, self.w), dtype=np.float32)
        
        
        for traffic_element in traffic_elements_info:
            # skip empty or non-sign elements
            if not traffic_element:
                continue
                
            if traffic_element['class'] != 'traffic_sign':
                continue
            
            type_id = traffic_element['type_id']
            is_stop = 'stop' in type_id
            is_speed_limit = 'speed_limit' in type_id
            
            # only process stop and speed_limit
            if not (is_stop or is_speed_limit):
                continue
            
            vertices_lidar = self._build_trigger_vertices_local(
                traffic_element['trigger_volume_location'],
                traffic_element['trigger_volume_size'],
                traffic_element['trigger_volume_rotation'],
                world2lidar,
            )
            trigger_mask = self._rasterize_polygon_mask(vertices_lidar)
            
            # stop sign
            if is_stop:
                # check if ego is inside trigger region
                if self.ego_in_bounds:
                    ego_in_trigger = trigger_mask[self.ego_row, self.ego_col]
                else:
                    ego_in_trigger = False
                
                if not ego_in_trigger:
                    # ego outside: higher speed → higher risk, larger extent
                    speed_factor = min(ego_vel / 30.0, 1.0)  # normalized speed factor (max 30 m/s)
                    
                    # compute extent and risk value
                    expand_meters = 5.0 * speed_factor  # higher speed → larger extent
                    expand_pixels = int(expand_meters * self.pixels_per_meter_x)
                    risk_value = speed_factor  # higher speed → higher risk
                    
                    # dilate to expand risk region
                    if expand_pixels > 0:
                        kernel = np.ones((expand_pixels * 2 + 1, expand_pixels * 2 + 1), np.uint8)
                        expanded_mask = cv2.dilate(trigger_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                    else:
                        expanded_mask = trigger_mask
                    
                    # set risk value
                    traffic_risk_field[expanded_mask] = np.maximum(
                        traffic_risk_field[expanded_mask], risk_value
                    )
                # risk is 0 once inside the trigger region
            
            # speed limit sign
            elif is_speed_limit:
                # extract speed limit from type_id
                try:
                    speed_limit = int(type_id.split('.')[-1])
                except (ValueError, IndexError):
                    continue
                
                # convert to m/s
                limit_ms = speed_limit / 3.6
                
                if ego_vel > limit_ms:
                    # overspeed: risk based on overspeed ratio
                    overspeed_ratio = (ego_vel - limit_ms) / limit_ms
                    speed_diff_factor = min(overspeed_ratio / 0.5, 1.0)
                    
                    # compute extent and risk value
                    expand_meters = 5.0 * speed_diff_factor  # more overspeed → larger extent
                    expand_pixels = int(expand_meters * self.pixels_per_meter_x)
                    risk_value = speed_diff_factor  # more overspeed → higher risk
                    
                    # dilate to expand risk region
                    if expand_pixels > 0:
                        kernel = np.ones((expand_pixels * 2 + 1, expand_pixels * 2 + 1), np.uint8)
                        expanded_mask = cv2.dilate(trigger_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                    else:
                        expanded_mask = trigger_mask
                    
                    # set risk value
                    traffic_risk_field[expanded_mask] = np.maximum(
                        traffic_risk_field[expanded_mask], risk_value
                    )
                # risk is 0 when not overspeeding
        
        return traffic_risk_field
    
    def generate_traffic_light_risk(self,
                                    traffic_elements_info: list,
                                    world2lidar: np.ndarray) -> np.ndarray:
        
        traffic_risk = np.zeros((self.h, self.w), dtype=np.float32)
        
        # risk value per traffic light state
        state_risk_map = {
            0: 0.9,    # red
            1: 0.5,    # yellow
            2: 0.0     # green
        }
        
        polygons_by_state = {0: [], 1: []}  # group polygon vertices by state: 0=red, 1=yellow
        
        for elem in traffic_elements_info:
            # skip empty elements
            if not elem:
                continue
                
            # only process traffic lights
            if elem['class'] != 'traffic_light':
                continue
            
            # get traffic light state and risk value
            state = elem['state']
            if state not in polygons_by_state:
                continue
            
            vertices_lidar = self._build_trigger_vertices_local(
                elem['trigger_volume_location'],
                elem['trigger_volume_size'],
                elem['trigger_volume_rotation'],
                world2lidar,
            )
            vertices_pixels = np.empty((vertices_lidar.shape[0], 2), dtype=np.float32)
            vertices_pixels[:, 0] = (vertices_lidar[:, 0] - self.x_min) * self.w / self.x_range
            vertices_pixels[:, 1] = (vertices_lidar[:, 1] - self.y_min) * self.h / self.y_range
            polygons_by_state[state].append(vertices_pixels.astype(np.int32))
            
        # process each state
        for state, polygons in polygons_by_state.items():
            if not polygons:
                continue
                
            risk_value = state_risk_map[state]
            
            # 1. create mask for all lights in this state
            
            state_mask = np.zeros((self.h, self.w), dtype=np.uint8)
            cv2.fillPoly(state_mask, polygons, 1)
            state_mask = state_mask.astype(bool)
            
            # 2. fill risk inside trigger region
            traffic_risk[state_mask] = np.maximum(traffic_risk[state_mask], risk_value)
            
            # 3. extend risk field outside trigger region
            expand_meters = 2.0 if state == 0 else 1.0
            expand_pixels = int(expand_meters * self.pixels_per_meter_x)
            
            # 4. dilate
            if expand_pixels > 0:
                kernel = np.ones((expand_pixels * 2 + 1, expand_pixels * 2 + 1), np.uint8)
                expanded_mask = cv2.dilate(state_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
                
                # 5. set decayed risk only in expanded region
                expand_only_mask = expanded_mask & ~state_mask
                if np.any(expand_only_mask):
                    # distance transform needs inverted mask
                    distance_field = cv2.distanceTransform((~state_mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                    
                    # apply decayed risk
                    traffic_risk[expand_only_mask] = np.maximum(
                        traffic_risk[expand_only_mask],
                        risk_value * np.exp(-distance_field[expand_only_mask] / expand_pixels)
                    )
        
        return traffic_risk
    

    def generate_object_risk_field(self,
                                   gt_boxes: np.ndarray,
                                   gt_names: list,
                                   gt_labels: np.ndarray,
                                   weather_factor: float) -> np.ndarray:
        # initialize risk field as float32
        risk_field = np.zeros(self.high_res_risk_size, dtype=np.float32)
        
        # object types that contribute risk
        target_classes = {'car', 'truck', 'bus', 'construction_vehicle', 'bicycle', 
                         'motorcycle', 'pedestrian', 'trailer', 'van', 'traffic_cone', 'others'}
        
        # filter objects that need risk
        valid_indices = [i for i, name in enumerate(gt_names) if name in target_classes]
        if not valid_indices:
            return risk_field
        
        # process detections
        filtered_boxes = gt_boxes[valid_indices]
        filtered_labels = gt_labels[valid_indices]
        
        # batch extract
        positions = filtered_boxes[:, :3]
        dimensions = filtered_boxes[:, 3:6]
        yaws = filtered_boxes[:, 6]
        velocities = filtered_boxes[:, 7:9]
        
        # batch compute speed
        speeds = np.linalg.norm(velocities[:, :2], axis=1)
        
        # fix heading: stored yaw is (-yaw_local - pi/2)
        # recover physical yaw_local (ego frame, x forward)
        # Inverse transform: yaw_local = -(yaw_saved + pi/2)
        yaws = -(yaws + np.pi / 2)
        
        # batch get risk weights
        risk_weights = np.array([self.object_risk_weights[int(label)] 
                                for label in filtered_labels])
        
        # batch compute speed expansion (log1p growth)
        # speed=0→1.0, speed=15m/s→≈1.72, speed=30m/s→≈1.96
        speed_expansions = 1.0 + 0.4 * np.log1p(speeds / 3.0)
        speed_expansions = np.clip(speed_expansions, 1.0, 2.5)

        # batch compute cos/sin
        cos_yaws = np.cos(yaws)
        sin_yaws = np.sin(yaws)
        
        # apply speed expansion, weather and risk weights
        # dimensions: [w, l, h]
        # sigma_cols (longitudinal) maps to length
        # sigma_rows (lateral) maps to width
        sigma_cols_m = dimensions[:, 1] * speed_expansions * weather_factor * risk_weights
        sigma_rows_m = dimensions[:, 0] * weather_factor * risk_weights
        
        sigma_cols = sigma_cols_m * self.pixels_per_meter_x
        sigma_rows = sigma_rows_m * self.pixels_per_meter_y
        
        # batch precompute BEV centers
        center_cols = (positions[:, 0] - self.x_min) * self.w / self.x_range
        center_rows = (positions[:, 1] - self.y_min) * self.h / self.y_range

        # per-object risk, merged into total field
        for i in range(len(filtered_boxes)):
            center_col = center_cols[i]
            center_row = center_rows[i]

            # bounds check
            if not (0 <= center_row < self.h and 0 <= center_col < self.w):
                continue
            # dimensions are [w, l, h]
            w_obj, l, h_obj = dimensions[i]

            # numerical stability: lower-bound sigma
            sigma_col = max(sigma_cols[i], self.min_sigma)
            sigma_row = max(sigma_rows[i], self.min_sigma)

            # influence radius (12σ longitudinal, 8σ lateral)
            influence_radius_col = int(16 * sigma_col)
            influence_radius_row = int(12 * sigma_row)
            
            # compute bounds
            start_row = max(0, int(center_row - influence_radius_row))
            end_row = min(self.h, int(center_row + influence_radius_row + 1))
            start_col = max(0, int(center_col - influence_radius_col))
            end_col = min(self.w, int(center_col + influence_radius_col + 1))
            
            # skip empty region
            if start_row >= end_row or start_col >= end_col:
                continue
            
            # local coordinate grid (row, col)
            y_coords_local, x_coords_local = np.ogrid[start_row:end_row, start_col:end_col]
            
            # coords relative to center
            dcol = x_coords_local - center_col  # x offset
            drow = y_coords_local - center_row  # y offset
            
            # rotate using precomputed cos/sin
            cos_yaw = cos_yaws[i]
            sin_yaw = sin_yaws[i]
            
            # rotated coords: longitudinal along heading, lateral perpendicular
            # Vector V = (dcol, drow) corresponding to (dx, dy)
            # Longitudinal = V dot Heading = dcol * cos + drow * sin
            # Lateral = V dot Perp_Heading = dcol * (-sin) + drow * cos
            
            dist_longitudinal = dcol * cos_yaw + drow * sin_yaw
            dist_lateral = -dcol * sin_yaw + drow * cos_yaw
            
            sigma_longitudinal = sigma_col  # Length direction
            sigma_lateral_val = sigma_row   # Width direction
            
            # Gaussian with rotated coords and adjusted sigma
            exponent = -0.5 * (
                (dist_longitudinal**2 / sigma_longitudinal**2) +  # along heading
                (dist_lateral**2 / sigma_lateral_val**2)    # perpendicular to heading
            )
            
            # clip exponent to prevent overflow
            exponent = np.clip(exponent, -self.exponent_clip_max, self.exponent_clip_max)
            
            # local risk field
            obj_risk_local = np.exp(exponent).astype(np.float32)

            risk_field[start_row:end_row, start_col:end_col] = self._merge_union(
                risk_field[start_row:end_row, start_col:end_col],
                obj_risk_local
            )

        return risk_field
    
    def _calculate_weather_factor(self, weather_info) -> float:
        # extract params, default 0.0
        wetness = weather_info['wetness']
        deposits = weather_info['precipitation_deposits']

        # normalize to 0~1
        wetness_norm = wetness / 100.0
        deposits_norm = deposits / 100.0

        # gentle growth: wetness sqrt, deposits square
        wetness_factor = 0.5 * np.sqrt(wetness_norm) + 1    # range [1, 1.5]
        deposits_factor = 0.5 * deposits_norm ** 2 + 1      # range [1, 1.5]

        # combine, keeping range [1, 1.5]
        weather_factor = np.sqrt(wetness_factor * deposits_factor)  # range [1, 1.5]
        
        return weather_factor
    
    def generate_risk_components(self,
                                 gt_boxes: np.ndarray,
                                 gt_names: List[str],
                                 gt_labels: np.ndarray,
                                 ego_vel: float,
                                 weather_info: Dict[str, Any],
                                 gt_masks_high: np.ndarray,
                                 gt_labels_high: np.ndarray,
                                 gt_bboxes_high: np.ndarray,
                                 traffic_elements_info: List[dict],
                                 world2lidar: np.ndarray,
                                 ) -> Dict[str, np.ndarray]:
        road_mask = self._create_road_boundary_mask(gt_masks_high, gt_labels_high, gt_bboxes_high)
        weather_factor = self._calculate_weather_factor(weather_info)

        object_risk_field = self.generate_object_risk_field(
            gt_boxes=gt_boxes,
            gt_names=gt_names,
            gt_labels=gt_labels,
            weather_factor=weather_factor
        )
        lane_risk_field = self.generate_lane_risk_field(
            gt_masks_high=gt_masks_high,
            gt_labels_high=gt_labels_high,
            gt_bboxes_high=gt_bboxes_high,
        )
        # add boundary risk to fill missing road boundaries
        boundary_risk_field = self.generate_road_boundary_risk_field(road_mask)
        lane_risk_field = self._merge_union(lane_risk_field, boundary_risk_field)
        traffic_light_risk_field = self.generate_traffic_light_risk(
            traffic_elements_info=traffic_elements_info,
            world2lidar=world2lidar,
        )
        traffic_sign_risk_field = self.generate_traffic_sign_risk_field(
            traffic_elements_info=traffic_elements_info,
            ego_vel=ego_vel,
            world2lidar=world2lidar,
        )

        other_risk_field = self._merge_union(traffic_light_risk_field, traffic_sign_risk_field)
        final_risk_field = self._merge_union(lane_risk_field, object_risk_field)
        final_risk_field = self._merge_union(final_risk_field, other_risk_field)

        final_risk_field = np.where(final_risk_field > self.risk_threshold, final_risk_field, 0.0)
        return {
            'final': np.clip(final_risk_field, 0.0, 1.0).astype(np.float32),
            'road_mask': road_mask.astype(np.float32),
        }

    def generate_unified_risk_field(self,
                                   gt_boxes: np.ndarray,
                                   gt_names: List[str],
                                   gt_labels: np.ndarray,
                                   ego_vel: float,
                                   weather_info: Dict[str, Any],
                                   gt_masks_high: np.ndarray,
                                   gt_labels_high: np.ndarray,
                                   gt_bboxes_high: np.ndarray,
                                   traffic_elements_info: List[dict],
                                   world2lidar: np.ndarray,
                                   ) -> Tuple[np.ndarray, np.ndarray]:
        components = self.generate_risk_components(
            gt_boxes=gt_boxes,
            gt_names=gt_names,
            gt_labels=gt_labels,
            ego_vel=ego_vel,
            weather_info=weather_info,
            gt_masks_high=gt_masks_high,
            gt_labels_high=gt_labels_high,
            gt_bboxes_high=gt_bboxes_high,
            traffic_elements_info=traffic_elements_info,
            world2lidar=world2lidar,
        )
        return components['final'], components['road_mask'].astype(bool)
    
