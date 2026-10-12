"""
CARLA closed-loop evaluation agent for Risk BEV Guide Planning model.

Follows the Bench2Drive agent interface (AutonomousAgent) and the patterns
established by UniAD/VAD reference agents.

Configuration: path_to_conf_file = "config.py+checkpoint.pth"
Example:
    TEAM_CONFIG=Bench2DriveZoo/adzoo/risk_guide_planning/configs/risk_bevformer/bevformer_riskformer_planner_b2d.py+Bench2DriveZoo/adzoo/risk_guide_planning/work_dirs/.../epoch_16.pth
"""
import os
import json
import pathlib
import time
import math
from collections import deque

import cv2
import carla
import numpy as np
import torch

from Bench2DriveZoo.team_code.pid_controller import PIDController
from Bench2DriveZoo.team_code.planner import RoutePlanner
from leaderboard.autoagents import autonomous_agent
from mmcv import Config
from mmcv.models import build_model
from mmcv.utils import load_checkpoint
from mmcv.core.bbox import get_box_type
from pyquaternion import Quaternion
from scipy.optimize import fsolve

SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)

# Planning frequency in CARLA ticks (1 tick = 0.05 s = 20 Hz).
# 1 = 20 Hz (infer every tick); 5 = 4 Hz; 10 = 2 Hz.
PLANNING_INTERVAL = 10


def get_entry_point():
    return 'RiskBeVGuidePlanningAgent'


class RiskBeVGuidePlanningAgent(autonomous_agent.AutonomousAgent):
    """CARLA evaluation agent for Risk BEV Guide Planning model.

    Loads a BEVFormerRiskPlanner model and runs closed-loop inference in CARLA.
    Converts model planning trajectories to vehicle controls via PID controller.
    """

    # ------------------------------------------------------------------
    #  Setup / Teardown
    # ------------------------------------------------------------------

    def setup(self, path_to_conf_file):
        """Load model, config, pipelines, and initialize controllers.

        path_to_conf_file format: "config_path+checkpoint_path[+save_name]"
        """
        self.track = autonomous_agent.Track.SENSORS
        self.step = -1
        self.initialized = False

        # Controllers
        self.pidcontroller = PIDController()

        # Parse config string
        parts = path_to_conf_file.split('+')
        self.config_path = parts[0]
        self.ckpt_path = parts[1]
        if IS_BENCH2DRIVE and len(parts) >= 3:
            self.save_name = parts[-1]
        else:
            self.save_name = 'risk_bev_guide'

        # Load mmcv config
        cfg = Config.fromfile(self.config_path)

        # Handle plugin imports (model registrations)
        if hasattr(cfg, 'plugin') and cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                # Handle absolute paths: extract the part from Bench2DriveZoo
                # onwards to form a valid dotted module path relative to CWD.
                if os.path.isabs(plugin_dir):
                    parts = plugin_dir.split('/')
                    try:
                        idx = parts.index('Bench2DriveZoo')
                        plugin_dir = '/'.join(parts[idx:])
                    except ValueError:
                        pass
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_dir = [m for m in _module_dir if m]  # filter empty strings
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                importlib.import_module(_module_path)

        # Build and load model
        self.model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'),
        )
        load_checkpoint(self.model, self.ckpt_path, map_location='cpu', strict=True)
        self.model.cuda()
        self.model.eval()

        # GPU preprocessing constants: NormalizeMultiviewImage(to_rgb=False)
        # mean=[103.53, 116.28, 123.675], std=[1,1,1] → subtract mean only
        # PadMultiViewImage(size_divisor=32, pad_val=0) → fill 0 in padded area
        self.img_mean = torch.tensor(
            [103.53, 116.28, 123.675], device='cuda', dtype=torch.float32).view(1, 1, 3, 1, 1)
        self.padded_H = 928   # ceil(900/32)*32
        self.padded_W = 1600  # ceil(1600/32)*32 = 1600

        # --- Hardcoded calibration (same sensor layout as all Bench2Drive agents) ---
        # lidar2img: LiDAR → image projection matrices (6 cameras, 4×4)
        self.lidar2img = {
            'CAM_FRONT': np.array([
                [1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -9.52000000e+02],
                [0.00000000e+00, 4.50000000e+02, -1.14251841e+03, -8.09704417e+02],
                [0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -1.19000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            ]),
            'CAM_FRONT_LEFT': np.array([
                [6.03961325e-14, 1.39475744e+03, 0.00000000e+00, -9.20539908e+02],
                [-3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                [-8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            ]),
            'CAM_FRONT_RIGHT': np.array([
                [1.31064327e+03, -4.77035138e+02, 0.00000000e+00, -4.06010608e+02],
                [3.68618420e+02, 2.58109396e+02, -1.14251841e+03, -6.47296750e+02],
                [8.19152044e-01, 5.73576436e-01, 0.00000000e+00, -8.29094072e-01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            ]),
            'CAM_BACK': np.array([
                [-5.60166031e+02, -8.00000000e+02, 0.00000000e+00, -1.28800000e+03],
                [5.51091060e-14, -4.50000000e+02, -5.60166031e+02, -8.58939847e+02],
                [1.22464680e-16, -1.00000000e+00, 0.00000000e+00, -1.61000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            ]),
            'CAM_BACK_LEFT': np.array([
                [-1.14251841e+03, 8.00000000e+02, 0.00000000e+00, -6.84385123e+02],
                [-4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                [-9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            ]),
            'CAM_BACK_RIGHT': np.array([
                [3.60989788e+02, -1.34723223e+03, 0.00000000e+00, -1.04238127e+02],
                [4.22861679e+02, -1.53909064e+02, -1.14251841e+03, -4.96004706e+02],
                [9.39692621e-01, -3.42020143e-01, 0.00000000e+00, -4.92889531e-01],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            ]),
        }

        # lidar2cam: LiDAR → camera extrinsic (4×4)
        self.lidar2cam = {
            'CAM_FRONT': np.array([
                [1., 0., 0., 0.],
                [0., 0., -1., -0.24],
                [0., 1., 0., -1.19],
                [0., 0., 0., 1.],
            ]),
            'CAM_FRONT_LEFT': np.array([
                [0.57357644, 0.81915204, 0., -0.22517331],
                [0., 0., -1., -0.24],
                [-0.81915204, 0.57357644, 0., -0.82909407],
                [0., 0., 0., 1.],
            ]),
            'CAM_FRONT_RIGHT': np.array([
                [0.57357644, -0.81915204, 0., 0.22517331],
                [0., 0., -1., -0.24],
                [0.81915204, 0.57357644, 0., -0.82909407],
                [0., 0., 0., 1.],
            ]),
            'CAM_BACK': np.array([
                [-1., 0., 0., 0.],
                [0., 0., -1., -0.24],
                [0., -1., 0., -1.61],
                [0., 0., 0., 1.],
            ]),
            'CAM_BACK_LEFT': np.array([
                [-0.34202014, 0.93969262, 0., -0.25388956],
                [0., 0., -1., -0.24],
                [-0.93969262, -0.34202014, 0., -0.49288953],
                [0., 0., 0., 1.],
            ]),
            'CAM_BACK_RIGHT': np.array([
                [-0.34202014, -0.93969262, 0., 0.25388956],
                [0., 0., -1., -0.24],
                [0.93969262, -0.34202014, 0., -0.49288953],
                [0., 0., 0., 1.],
            ]),
        }

        # lidar2ego: LiDAR → ego-vehicle transform
        self.lidar2ego = np.array([
            [0., 1., 0., -0.39],
            [-1., 0., 0., 0.],
            [0., 0., 1., 1.84],
            [0., 0., 0., 1.],
        ])

        # GPS reference (initialized in _init)
        self.lat_ref = 42.0
        self.lon_ref = 2.0

        # Temporal state for prev_bev management (matrix-based, matching union2one)
        self.prev_bev = None
        self.prev_world2lidar = None
        self.last_traj_world = None   # cached trajectory (world frame) for PID between inference frames

        # History buffers for ego_his_trajs and ego_lcf_feat
        self.pos_history = deque(maxlen=5)    # world-frame positions (most recent last)
        self.yaw_history = deque(maxlen=5)    # yaw angles in radians

        # Control history for ego_lcf_feat (steer from previous frame)
        self.prev_control = carla.VehicleControl()
        self.prev_control.steer = 0.0
        self.prev_control.throttle = 0.0
        self.prev_control.brake = 0.0

        # Save path
        self.save_path = None
        if SAVE_PATH is not None:
            string = self.save_name
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)

        self.metric_info = {}

    def destroy(self):
        del self.model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    #  Sensors definition
    # ------------------------------------------------------------------

    def sensors(self):
        sensors = [
            {
                'type': 'sensor.camera.rgb',
                'x': 0.80, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT',
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_LEFT',
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.27, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_FRONT_RIGHT',
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -2.0, 'y': 0.0, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                'width': 1600, 'height': 900, 'fov': 110,
                'id': 'CAM_BACK',
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': -0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_LEFT',
            },
            {
                'type': 'sensor.camera.rgb',
                'x': -0.32, 'y': 0.55, 'z': 1.60,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                'width': 1600, 'height': 900, 'fov': 70,
                'id': 'CAM_BACK_RIGHT',
            },
            {
                'type': 'sensor.other.imu',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'IMU',
            },
            {
                'type': 'sensor.other.gnss',
                'x': -1.4, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'GPS',
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'SPEED',
            },
        ]
        if IS_BENCH2DRIVE:
            sensors.append({
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 50.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 512, 'height': 512, 'fov': 5 * 10.0,
                'id': 'bev',
            })
        return sensors

    # ------------------------------------------------------------------
    #  GPS / coordinate utilities
    # ------------------------------------------------------------------

    def _init(self):
        """Initialize GPS reference and route planner from global plan."""
        try:
            locx = self._global_plan_world_coord[0][0].location.x
            locy = self._global_plan_world_coord[0][0].location.y
            lon = self._global_plan[0][0]['lon']
            lat = self._global_plan[0][0]['lat']
            EARTH_RADIUS_EQUA = 6378137.0

            def equations(vars):
                x, y = vars
                eq1 = (lon * math.cos(x * math.pi / 180)
                       - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA)
                       - math.cos(x * math.pi / 180) * y)
                eq2 = (math.log(math.tan((lat + 90) * math.pi / 360))
                       * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180)
                       + locy
                       - math.cos(x * math.pi / 180)
                       * EARTH_RADIUS_EQUA
                       * math.log(math.tan((90 + x) * math.pi / 360)))
                return [eq1, eq2]

            solution = fsolve(equations, [0, 0])
            self.lat_ref, self.lon_ref = float(solution[0]), float(solution[1])
        except Exception:
            self.lat_ref, self.lon_ref = 0.0, 0.0

        self._route_planner = RoutePlanner(
            4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref,
        )
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def gps_to_location(self, gps):
        """Convert GPS (lat, lon) to world coordinates via Mercator projection."""
        EARTH_RADIUS_EQUA = 6378137.0
        lat, lon = gps
        scale = math.cos(self.lat_ref * math.pi / 180.0)
        my = (math.log(math.tan((lat + 90) * math.pi / 360.0))
              * (EARTH_RADIUS_EQUA * scale))
        mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
        y = (scale * EARTH_RADIUS_EQUA
             * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0))
             - my)
        x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
        return np.array([x, y])

    # ------------------------------------------------------------------
    #  Sensor preprocessing ('tick')
    # ------------------------------------------------------------------

    def tick(self, input_data):
        """Preprocess raw CARLA sensor data into model-ready format."""
        self.step += 1

        # Extract and process camera images
        # CARLA outputs BGRA; strip alpha → BGR (model uses to_rgb=False)
        imgs = {}
        for cam in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            img = input_data[cam][1][:, :, :3]  # BGRA → BGR
            imgs[cam] = img

        # BEV camera (Bench2Drive mode)
        bev = None
        if IS_BENCH2DRIVE and 'bev' in input_data:
            bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)

        # Extract other sensors
        gps = input_data['GPS'][1][:2]
        speed = input_data['SPEED'][1]['speed']
        compass = input_data['IMU'][1][-1]
        acceleration = input_data['IMU'][1][:3]
        angular_velocity = input_data['IMU'][1][3:6]

        # Handle NaN compass (known CARLA issue)
        if math.isnan(compass):
            compass = 0.0
            acceleration = np.zeros(3)
            angular_velocity = np.zeros(3)

        # GPS → world coordinates
        pos = self.gps_to_location(gps)

        # Route planning
        near_node, near_command = self._route_planner.run_step(pos)

        return {
            'imgs': imgs,
            'gps': gps,
            'pos': pos,
            'speed': speed,
            'compass': compass,
            'bev': bev,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'command_near': near_command,
            'command_near_xy': near_node,
        }

    # ------------------------------------------------------------------
    #  Main inference loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        """Run one inference step: preprocess → model → PID → control.

        Model inference runs at 2 Hz (every 10th CARLA tick, ~0.5 s interval).
        Training data was collected at 10 Hz with sample_interval=5, giving a
        0.5 s temporal gap between consecutive queue frames.  CARLA runs at
        20 Hz, so 10 ticks = 0.5 s matches the training temporal design.
        Between inference frames the last trajectory is reused — PID naturally
        tracks forward along it as the vehicle moves.
        """
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        # record ego state every tick (20 Hz) for efficiency/smoothness metrics
        self.metric_info[self.step] = self.get_metric_info()

        # ---- Per-frame essentials (always needed for PID + history) ----

        can_bus = self._build_can_bus(tick_data)

        ego_theta = -tick_data['compass'] + np.pi / 2
        if math.isnan(tick_data['compass']):
            ego_theta = 0.0

        command = tick_data['command_near']
        if command < 0:
            command = 4
        command -= 1  # 1-6 → 0-5

        local_command_xy = self._compute_local_command(tick_data, can_bus)

        # Record ego history at a fixed 2 Hz (every 10 ticks), decoupled from the
        # inference frequency.  _build_ego_his_trajs stores step displacements whose
        # magnitude is proportional to the recording interval.  Training uses a 0.5 s
        # (2 Hz) interval, so recording every 10 ticks keeps displacement semantics
        # consistent with training.
        if self.step % 10 == 0:
            self.pos_history.append(tick_data['pos'])
            self.yaw_history.append(ego_theta)

        # ---- Frame skip: reuse the cached trajectory, PID converts to current ego frame ----
        if self.step % PLANNING_INTERVAL != 0 and self.last_traj_world is not None:
            world2lidar_cur = self._compute_world2lidar(can_bus[:3], ego_theta)
            traj_cur = self._world_to_lidar_xy(self.last_traj_world, world2lidar_cur)
            steer, throttle, brake, metadata = self.pidcontroller.control_pid(
                traj_cur, tick_data['speed'], local_command_xy,
            )
            if brake < 0.05:
                brake = 0.0
            if throttle > brake:
                brake = 0.0
            control = carla.VehicleControl()
            control.steer = float(np.clip(steer, -1.0, 1.0))
            control.throttle = float(np.clip(throttle, 0.0, 0.75))
            control.brake = float(np.clip(brake, 0.0, 1.0))
            self.pid_metadata = metadata
            self.prev_control = control
            return control

        # ===== Full model inference =====

        # 1. Images: GPU preprocessing (Normalize + Pad to 32x)
        cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        img_tensor = torch.zeros(1, 6, 3, self.padded_H, self.padded_W,
                                 device='cuda', dtype=torch.float32)
        for i, cam in enumerate(cam_order):
            img = tick_data['imgs'][cam]
            t = torch.from_numpy(img).cuda().permute(2, 0, 1).float()
            img_tensor[0, i, :, :900, :] = t - self.img_mean.view(3, 1, 1)

        # 2. world2lidar from current ego pose
        world2lidar = self._compute_world2lidar(can_bus[:3], ego_theta)

        # 3. img_metas
        img_meta = self._build_img_meta(tick_data, can_bus,
                                        (self.padded_H, self.padded_W, 3), world2lidar)

        # 4. Temporal: relative can_bus matching union2one (matrix-based)
        can_bus_rel = can_bus.copy()
        if self.prev_bev is not None:
            prev_from_cur = self.prev_world2lidar @ np.linalg.inv(world2lidar)
            delta_yaw = np.arctan2(prev_from_cur[1, 0], prev_from_cur[0, 0])
            can_bus_rel[:3] = prev_from_cur[:3, 3]
            can_bus_rel[16] = delta_yaw
            can_bus_rel[17] = delta_yaw / np.pi * 180
        else:
            can_bus_rel[:3] = 0
            can_bus_rel[16] = 0
            can_bus_rel[17] = 0
        img_meta['can_bus'] = can_bus_rel.tolist()

        # Store world2lidar for next inference frame
        self.prev_world2lidar = world2lidar

        # 5. Navigation goal_point: far-away route node (~35 m ahead) in LiDAR frame.
        #    Walk forward along the route until cumulative distance reaches the
        #    target range.  A far goal (vs ~4 m near point) gives the scoring head
        #    meaningful ability to distinguish on-track vs off-track trajectories.
        goal_xy_world = self._get_far_route_point(tick_data['pos'], world2lidar)
        # goal_xy_world is in Mercator coords [x, y] (y not negated), while world2lidar
        # uses the CARLA world convention (y negated, see can_bus[1] = -pos[1]).  Convert
        # to CARLA world before projecting, otherwise the goal distance is inflated.
        goal_xy_carla = np.array([goal_xy_world[0], -goal_xy_world[1]], dtype=np.float32)
        goal_point = np.array([
            world2lidar[0, 0] * goal_xy_carla[0] + world2lidar[0, 1] * goal_xy_carla[1] + world2lidar[0, 3],
            world2lidar[1, 0] * goal_xy_carla[0] + world2lidar[1, 1] * goal_xy_carla[1] + world2lidar[1, 3],
        ], dtype=np.float32)

        # 6. ego_his_trajs and ego_lcf_feat
        ego_his_trajs, ego_his_trajs_mask = self._build_ego_his_trajs()
        ego_lcf_feat = self._build_ego_lcf_feat(tick_data, can_bus)

        # ---- Model inference ----
        new_bev, results = self.model.simple_test(
            [img_meta],
            img_tensor,
            prev_bev=self.prev_bev,
            command=torch.as_tensor([command], device='cuda'),
            ego_his_trajs=torch.as_tensor(ego_his_trajs, device='cuda', dtype=torch.float32).unsqueeze(0),
            ego_his_trajs_mask=torch.as_tensor(ego_his_trajs_mask, device='cuda', dtype=torch.float32).unsqueeze(0),
            ego_lcf_feat=torch.as_tensor(ego_lcf_feat, device='cuda', dtype=torch.float32).unsqueeze(0),
            goal_point=torch.as_tensor(goal_point, device='cuda', dtype=torch.float32).unsqueeze(0),
        )

        # Update temporal state
        self.prev_bev = new_bev

        # ---- Extract trajectory and control ----
        planning_traj = results[0]['planning_traj']  # (6, 3)

        # cache trajectory to world frame (absolute coords) for later PID conversion
        lidar2world = np.linalg.inv(world2lidar)
        self.last_traj_world = self._lidar_to_world_xy(planning_traj[:, :2], lidar2world)

        out_truck = planning_traj[:, :2]              # (6, 2)

        steer, throttle, brake, metadata = self.pidcontroller.control_pid(
            out_truck, tick_data['speed'], local_command_xy,
        )

        if brake < 0.05:
            brake = 0.0
        if throttle > brake:
            brake = 0.0

        control = carla.VehicleControl()
        control.steer = float(np.clip(steer, -1.0, 1.0))
        control.throttle = float(np.clip(throttle, 0.0, 0.75))
        control.brake = float(np.clip(brake, 0.0, 1.0))

        # Save metadata for debugging
        self.pid_metadata = metadata
        self.pid_metadata['agent'] = 'only_traj'
        self.pid_metadata['steer'] = control.steer
        self.pid_metadata['throttle'] = control.throttle
        self.pid_metadata['brake'] = control.brake
        self.pid_metadata['plan'] = out_truck.tolist()
        self.pid_metadata['command'] = command
        self.pid_metadata['steer_traj'] = float(steer)
        self.pid_metadata['throttle_traj'] = float(throttle)
        self.pid_metadata['brake_traj'] = float(brake)

        if SAVE_PATH is not None and self.step % 10 == 0:
            self._save(tick_data)

        self.prev_control = control
        return control

    # ------------------------------------------------------------------
    #  Model input builders
    # ------------------------------------------------------------------

    def _compute_world2lidar(self, ego_translation, ego_theta):
        """Compute world-to-LiDAR transform consistent with the dataset convention.

        Args:
            ego_translation: [x, -y, 0] in dataset world frame (from can_bus[:3])
            ego_theta: ego yaw angle in radians (= -compass + pi/2)

        Returns:
            4×4 world2lidar matrix
        """
        rotation = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        ego2world = np.eye(4, dtype=np.float64)
        ego2world[:3, :3] = rotation
        ego2world[:3, 3] = ego_translation
        lidar2world = ego2world @ self.lidar2ego
        return np.linalg.inv(lidar2world)

    @staticmethod
    def _lidar_to_world_xy(pts, lidar2world):
        """Convert LiDAR-frame xy points (N,2) to world frame."""
        R = lidar2world[:2, :2]
        t = lidar2world[:2, 3]
        return (R @ pts.T).T + t

    @staticmethod
    def _world_to_lidar_xy(pts, world2lidar):
        """Convert world-frame xy points (N,2) to LiDAR frame."""
        R = world2lidar[:2, :2]
        t = world2lidar[:2, 3]
        return (R @ pts.T).T + t

    def _get_far_route_point(self, ego_world, world2lidar, target_distance=25.0):
        """Return a point on the route polyline exactly target_distance ahead of ego.

        Projects ego onto the current segment route[0]→route[1] (RoutePlanner's
        run_step already pops passed nodes monotonically, so this segment is where
        the ego currently is) and walks target_distance forward along the route.

        Using only the current segment — instead of searching the whole polyline
        for the nearest segment — keeps the ego's arc-length position (and hence
        the goal) stable across frames: it only advances when a node is popped,
        rather than jumping between segments of a sparse polyline during lane changes.
        """
        route = self._route_planner.route
        if len(route) < 2:
            # fallback: use the near command point
            return self._route_planner.route[0][0]

        # project ego onto current segment route[0]→route[1] (both Mercator coords)
        p0 = route[0][0]
        p1 = route[1][0]
        seg = p1 - p0
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            return p1
        t = np.clip(np.dot(ego_world - p0, seg) / (seg_len ** 2), 0.0, 1.0)
        proj = p0 + t * seg

        # advance target_distance from the projection along the route
        cumulative = (1.0 - t) * seg_len  # distance from projection to end of current segment (route[1])
        prev = proj
        if cumulative >= target_distance:
            return prev + (target_distance / seg_len) * seg

        for idx in range(1, len(route) - 1):
            node = route[idx][0]
            next_node = route[idx + 1][0]
            seg2 = next_node - node
            seg2_len = np.linalg.norm(seg2)
            if cumulative + seg2_len >= target_distance:
                frac = (target_distance - cumulative) / max(seg2_len, 1e-6)
                return node + frac * seg2
            cumulative += seg2_len

        # Route exhausted before reaching target_distance — return last node
        return route[-1][0]

    def _build_can_bus(self, tick_data):
        """Build 18-dim can_bus vector from sensor data."""
        ego_theta = -tick_data['compass'] + np.pi / 2
        if math.isnan(tick_data['compass']):
            ego_theta = 0.0
        rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))

        can_bus = np.zeros(18, dtype=np.float32)
        can_bus[0] = tick_data['pos'][0]
        can_bus[1] = -tick_data['pos'][1]
        can_bus[3:7] = rotation
        can_bus[7] = tick_data['speed']
        can_bus[10:13] = tick_data['acceleration']
        can_bus[11] *= -1
        can_bus[13:16] = -tick_data['angular_velocity']
        can_bus[16] = ego_theta
        can_bus[17] = ego_theta / np.pi * 180
        return can_bus

    def _build_img_meta(self, tick_data, can_bus, img_shape, world2lidar):
        """Build img_meta dict matching BEVFormer meta keys."""
        cam_order = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        lidar2img = np.stack([self.lidar2img[c] for c in cam_order], axis=0)
        lidar2cam = np.stack([self.lidar2cam[c] for c in cam_order], axis=0)

        meta = {
            'can_bus': can_bus,
            'world2lidar': world2lidar.astype(np.float32),
            'lidar2img': lidar2img,
            'lidar2cam': lidar2cam,
            'folder': ' ',
            'scene_token': ' ',
            'frame_idx': 0,
            'timestamp': self.step / 20,
            'img_shape': [(img_shape[0], img_shape[1], 3)] * 6,
            'ori_shape': [(900, 1600, 3)] * 6,
            'pad_shape': [(img_shape[0], img_shape[1], 3)] * 6,
        }
        return meta

    def _build_ego_his_trajs(self):
        """Build ego history trajectories (4 past displacements in LiDAR frame).

        Returns (ego_his_trajs, ego_his_trajs_mask) as numpy arrays.
        ego_his_trajs: (4, 2) float32, displacements between consecutive history frames
        ego_his_trajs_mask: (4, 2) float32, 1.0 for valid, 0.0 for padded
        """
        n_hist = len(self.pos_history)
        ego_his_trajs = np.zeros((4, 2), dtype=np.float32)
        ego_his_trajs_mask = np.zeros((4, 2), dtype=np.float32)

        if n_hist < 2:
            return ego_his_trajs, ego_his_trajs_mask

        # Compute world→lidar rotation for current frame (the latest yaw)
        ego_theta = self.yaw_history[-1]
        ego2world_rot = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
        lidar2ego_rot = self.lidar2ego[:3, :3]
        lidar2world_rot = ego2world_rot @ lidar2ego_rot
        world2lidar_rot = np.linalg.inv(lidar2world_rot)

        positions = list(self.pos_history)
        n_disp = n_hist - 1  # number of displacements we have
        n_needed = 4

        # Latest displacements go to higher indices
        start_idx = n_needed - min(n_disp, n_needed)
        for i in range(min(n_disp, n_needed)):
            idx = start_idx + i
            p_prev = positions[i]
            p_next = positions[i + 1]
            disp_world = np.array([p_next[0] - p_prev[0],
                                   p_next[1] - p_prev[1],
                                   0.0])
            disp_lidar = world2lidar_rot @ disp_world
            ego_his_trajs[idx] = disp_lidar[:2].astype(np.float32)
            ego_his_trajs_mask[idx] = 1.0

        return ego_his_trajs, ego_his_trajs_mask

    def _build_ego_lcf_feat(self, tick_data, can_bus):
        """Build 9-dim ego local coordinate feature matching dataset format.

        Index mapping (from B2D_Risk_Dataset):
          [0:2] = ego_translation (x, y)
          [2:4] = ego_accel (x, y)
          [4]   = ego_rotation_rate[-1] (yaw rate)
          [5]   = ego_size[1] = vehicle LENGTH  (4.8924)
          [6]   = ego_size[0] = vehicle WIDTH   (1.8367)
          [7]   = speed (magnitude of velocity)
          [8]   = steer
        """
        ego_lcf_feat = np.zeros(9, dtype=np.float32)
        ego_lcf_feat[0:2] = can_bus[0:2].astype(np.float32)
        # Match dataset convention: ego_accel = [accel_x, -accel_y, accel_z]
        ego_lcf_feat[2] = float(tick_data['acceleration'][0])
        ego_lcf_feat[3] = -float(tick_data['acceleration'][1])
        # Match dataset convention: ego_rotation_rate = -angular_velocity
        ego_lcf_feat[4] = -float(tick_data['angular_velocity'][-1])
        ego_lcf_feat[5] = 4.8924   # vehicle length (ego_size[1])
        ego_lcf_feat[6] = 1.8367   # vehicle width  (ego_size[0])
        ego_lcf_feat[7] = float(tick_data['speed'])
        ego_lcf_feat[8] = float(self.prev_control.steer)
        return ego_lcf_feat

    def _compute_local_command(self, tick_data, can_bus):
        """Compute route target in LiDAR frame for PID controller."""
        raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0.0
        command_near_xy = np.array([
            tick_data['command_near_xy'][0] - can_bus[0],
            -tick_data['command_near_xy'][1] - can_bus[1],
        ])
        rotation_matrix = np.array([
            [np.cos(raw_theta), -np.sin(raw_theta)],
            [np.sin(raw_theta), np.cos(raw_theta)],
        ])
        return rotation_matrix @ command_near_xy

    # ------------------------------------------------------------------
    #  Save / visualization
    # ------------------------------------------------------------------

    def _save(self, tick_data):
        """Save metrics to disk."""
        with open(self.save_path / 'metric_info.json', 'w') as f:
            json.dump(self.metric_info, f, indent=4)
