import numpy as np
import torch
from mmcv.models import DETECTORS, build_head
from mmcv.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmcv.models.utils.grid_mask import GridMask
from mmcv.utils import auto_fp16


@DETECTORS.register_module()
class BEVFormerRiskPlanner(MVXTwoStageDetector):
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 risk_cognition_head=None,
                 planning_head=None,
                 freeze_bev=True,
                 freeze_backbone=True,
                 **kwargs):
        super().__init__(
            pts_voxel_layer, pts_voxel_encoder, pts_middle_encoder,
            pts_fusion_layer, img_backbone, pts_backbone, img_neck,
            pts_neck, pts_bbox_head, img_roi_head, img_rpn_head,
            train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_world2lidar': None,
        }
        self.risk_cognition_head = build_head(risk_cognition_head)
        self.planning_head = build_head(planning_head)
        self.freeze_bev = freeze_bev
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            self._set_backbone_trainable(False)
        if self.freeze_bev:
            self._set_bev_head_trainable(False)

    def _set_backbone_trainable(self, trainable):
        """Control img_backbone + img_neck only (excludes pts_bbox_head)."""
        for module in (self.img_backbone, self.img_neck):
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = trainable
            if not trainable:
                module.eval()

    def _set_bev_head_trainable(self, trainable):
        """Control pts_bbox_head (BEVFormer) independently."""
        if self.pts_bbox_head is not None:
            for param in self.pts_bbox_head.parameters():
                param.requires_grad = trainable
            if not trainable:
                self.pts_bbox_head.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self._set_backbone_trainable(not self.freeze_backbone)
            self._set_bev_head_trainable(not self.freeze_bev)
        return self

    @staticmethod
    def _stack_batch_field(field, device=None, dtype=None):
        if hasattr(field, 'data'):
            field = field.data
        if isinstance(field, torch.Tensor):
            tensor = field
        elif isinstance(field, list):
            values = [item.data if hasattr(item, 'data') else item for item in field]
            tensor = torch.stack(values) if all(isinstance(item, torch.Tensor) for item in values) else torch.as_tensor(values)
        else:
            tensor = torch.as_tensor(field)
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device, dtype=dtype)
        return tensor

    @staticmethod
    def _extract_data(value):
        return value.data if hasattr(value, 'data') else value

    @staticmethod
    def _unwrap_single_test_aug(value):
        if isinstance(value, list):
            return value[0]
        return value

    @staticmethod
    def _normalize_test_field(value, squeeze_last=False, dtype=None, device=None):
        value = BEVFormerRiskPlanner._unwrap_single_test_aug(value)
        tensor = BEVFormerRiskPlanner._stack_batch_field(value, device=device, dtype=dtype)
        if squeeze_last:
            while tensor.dim() > 1 and tensor.shape[-1] == 1:
                tensor = tensor.squeeze(-1)
        return tensor

    def _extract_can_bus(self, img_metas, device):
        can_bus = np.asarray([meta['can_bus'] for meta in img_metas], dtype=np.float32)
        can_bus = torch.as_tensor(can_bus, device=device)
        assert can_bus.shape == (len(img_metas), 18), f'can_bus must be [B,18], got {can_bus.shape}'
        return can_bus

    def extract_img_feat(self, img, img_metas):
        B = img.size(0)
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None):
        return self.extract_img_feat(img, img_metas)

    def extract_bev(self, img, img_metas, prev_bev=None):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return self.pts_bbox_head(img_feats, img_metas, prev_bev=prev_bev, only_bev=True)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Compute prev_bev from historical frames iteratively.

        Iterates over historical frames in the queue, computing BEV features
        sequentially for temporal self-attention alignment.  Gradients never
        flow through history — the loop would be prohibitively expensive and
        the gradient signal through multiple past frames is uninformative.

        Args:
            imgs_queue: [B, Q_hist, N_cam, C, H, W] historical frame images
            img_metas_list: list of lists, each inner list has Q meta dicts
                (one per queue frame, including the current frame at [-1])

        Returns:
            prev_bev: BEV features from the last historical frame, or None
        """
        _, len_queue = imgs_queue.shape[:2]
        prev_bev = None
        with torch.no_grad():
            for i in range(len_queue):
                img = imgs_queue[:, i, ...]                       # [B, N_cam, C, H, W]
                img_metas = [each[i] for each in img_metas_list]   # meta for frame i
                if not img_metas[0]['prev_bev_exists']:
                    prev_bev = None
                prev_bev = self.extract_bev(img, img_metas, prev_bev=prev_bev)
        return prev_bev

    def forward(self, inputs, return_loss=True, rescale=False):
        if return_loss:
            losses = self.forward_train(**inputs)
            loss, log_vars = self._parse_losses(losses)
            return dict(loss=loss, log_vars=log_vars, num_samples=len(inputs['img_metas']))
        return self.forward_test(**inputs, rescale=rescale)

    @auto_fp16(apply_to=('img', 'prev_bev'))
    def forward_train(self,
                      img_metas=None,
                      img=None,
                      gt_risk_score=None,
                      gt_risk_valid_mask=None,
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      ego_his_trajs=None,
                      ego_his_trajs_mask=None,
                      ego_lcf_feat=None,
                      goal_point=None,
                      prev_bev=None):
        img = self._extract_data(img)
        img_metas = self._extract_data(img_metas)
        device = next(self.parameters()).device
        img = img.to(device)

        # Temporal: split queue-stacked images, compute prev_bev from history
        if img.dim() == 6 and img.size(1) > 1:
            prev_img = img[:, :-1, ...]             # [B, Q-1, N_cam, C, H, W]
            img = img[:, -1, ...]                    # [B, N_cam, C, H, W]
            prev_bev = self.obtain_history_bev(prev_img, img_metas)
            img_metas = [each[-1] for each in img_metas]
            if not img_metas[0]['prev_bev_exists']:
                prev_bev = None

        if self.freeze_backbone and self.freeze_bev:
            with torch.no_grad():
                bev_embed = self.extract_bev(img, img_metas, prev_bev=prev_bev)
        else:
            bev_embed = self.extract_bev(img, img_metas, prev_bev=prev_bev)

        can_bus = self._extract_can_bus(img_metas, bev_embed.device)
        losses = {}

        risk_outputs = self.risk_cognition_head(bev_embed, can_bus=can_bus)
        gt_risk_score_tensor = self._stack_batch_field(gt_risk_score, device=device, dtype=torch.float32)
        gt_risk_valid_mask_tensor = self._stack_batch_field(gt_risk_valid_mask, device=device, dtype=torch.float32)
        losses.update(self.risk_cognition_head.loss(risk_outputs, gt_risk_score_tensor, gt_risk_valid_mask_tensor))

        sdc_planning_tensor = self._stack_batch_field(sdc_planning, device=device, dtype=torch.float32)
        sdc_planning_mask_tensor = self._stack_batch_field(sdc_planning_mask, device=device, dtype=torch.float32)
        command_tensor = self._stack_batch_field(command, device=device).squeeze(-1)
        ego_his_trajs_tensor = self._stack_batch_field(ego_his_trajs, device=device, dtype=torch.float32)
        ego_his_trajs_mask_tensor = self._stack_batch_field(ego_his_trajs_mask, device=device, dtype=torch.float32)
        ego_lcf_feat_tensor = self._stack_batch_field(ego_lcf_feat, device=device, dtype=torch.float32)
        goal_point_tensor = self._stack_batch_field(goal_point, device=device, dtype=torch.float32)
        planning_outputs = self.planning_head(
            bev_embed=bev_embed,
            risk_map=risk_outputs['final'].detach().clamp(0.0, 1.0),
            gt_risk_map=gt_risk_score_tensor.detach(),
            command=command_tensor,
            can_bus=can_bus,
            ego_his_trajs=ego_his_trajs_tensor,
            ego_his_trajs_mask=ego_his_trajs_mask_tensor,
            ego_lcf_feat=ego_lcf_feat_tensor,
            goal_point=goal_point_tensor,
            sdc_planning=sdc_planning_tensor,
        )
        losses.update(self.planning_head.loss(
            planning_outputs,
            sdc_planning_tensor,
            sdc_planning_mask_tensor,
        ))
        return losses

    def val_step(self, data, optimizer=None):
        img = self._extract_data(data['img'])
        img_metas = self._extract_data(data['img_metas'])
        device = next(self.parameters()).device
        img = img.to(device)
        prev_bev = data.get('prev_bev', None)
        return self.extract_bev(img, img_metas, prev_bev=prev_bev)

    def forward_test(self, img_metas, img=None, rescale=None, **kwargs):
        if not isinstance(img_metas, list):
            raise TypeError(f'img_metas must be a list, got {type(img_metas)}')
        img = [img] if img is None else img
        gt_risk_score = kwargs.get('gt_risk_score', None)
        gt_risk_valid_mask = kwargs.get('gt_risk_valid_mask', None)
        sdc_planning = kwargs.get('sdc_planning', None)
        sdc_planning_mask = kwargs.get('sdc_planning_mask', None)
        command = kwargs.get('command', None)
        ego_his_trajs = kwargs.get('ego_his_trajs', None)
        ego_his_trajs_mask = kwargs.get('ego_his_trajs_mask', None)
        ego_lcf_feat = kwargs.get('ego_lcf_feat', None)
        goal_point = kwargs['goal_point']
        cur_meta = img_metas[0][0]
        if cur_meta['scene_token'] != self.prev_frame_info['scene_token']:
            self.prev_frame_info['prev_bev'] = None
        self.prev_frame_info['scene_token'] = cur_meta['scene_token']
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        cur_world2lidar = np.asarray(cur_meta['world2lidar'], dtype=np.float32)
        if self.prev_frame_info['prev_bev'] is not None:
            prev_from_cur = self.prev_frame_info['prev_world2lidar'] @ np.linalg.inv(cur_world2lidar)
            delta_yaw = np.arctan2(prev_from_cur[1, 0], prev_from_cur[0, 0])
            cur_meta['can_bus'][:3] = prev_from_cur[:3, 3]
            cur_meta['can_bus'][16] = delta_yaw
            cur_meta['can_bus'][17] = delta_yaw / np.pi * 180
        else:
            cur_meta['can_bus'][:3] = 0
            cur_meta['can_bus'][16] = 0
            cur_meta['can_bus'][17] = 0

        gt_risk_score_data = self._extract_data(gt_risk_score) if gt_risk_score is not None else None
        gt_risk_valid_mask_data = self._extract_data(gt_risk_valid_mask) if gt_risk_valid_mask is not None else None
        sdc_planning_data = self._extract_data(sdc_planning) if sdc_planning is not None else None
        sdc_planning_mask_data = self._extract_data(sdc_planning_mask) if sdc_planning_mask is not None else None
        command_data = self._extract_data(command) if command is not None else None
        ego_his_trajs_data = self._extract_data(ego_his_trajs) if ego_his_trajs is not None else None
        ego_his_trajs_mask_data = self._extract_data(ego_his_trajs_mask) if ego_his_trajs_mask is not None else None
        ego_lcf_feat_data = self._extract_data(ego_lcf_feat) if ego_lcf_feat is not None else None
        goal_point_data = self._extract_data(goal_point)
        new_prev_bev, results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'],
            rescale=rescale,
            gt_risk_score=gt_risk_score_data,
            gt_risk_valid_mask=gt_risk_valid_mask_data,
            sdc_planning=sdc_planning_data,
            sdc_planning_mask=sdc_planning_mask_data,
            command=command_data,
            ego_his_trajs=ego_his_trajs_data,
            ego_his_trajs_mask=ego_his_trajs_mask_data,
            ego_lcf_feat=ego_lcf_feat_data,
            goal_point=goal_point_data)
        self.prev_frame_info['prev_world2lidar'] = cur_world2lidar
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False, gt_risk_score=None, gt_risk_valid_mask=None, sdc_planning=None,
                    sdc_planning_mask=None, command=None, ego_his_trajs=None, ego_his_trajs_mask=None, ego_lcf_feat=None, goal_point=None):
        device = next(self.parameters()).device
        img = img.to(device)
        bev_embed = self.extract_bev(img, img_metas, prev_bev=prev_bev)
        can_bus = self._extract_can_bus(img_metas, bev_embed.device)
        risk_outputs = self.risk_cognition_head(bev_embed, can_bus=can_bus)
        assert command is not None, 'command is required for planning test'
        assert ego_his_trajs is not None, 'ego_his_trajs is required for planning test'
        assert ego_his_trajs_mask is not None, 'ego_his_trajs_mask is required for planning test'
        assert ego_lcf_feat is not None, 'ego_lcf_feat is required for planning test'
        gt_risk_score_tensor = self._normalize_test_field(gt_risk_score, dtype=torch.float32, device=bev_embed.device) if gt_risk_score is not None else None
        gt_risk_valid_mask_tensor = self._normalize_test_field(gt_risk_valid_mask, dtype=torch.float32, device=bev_embed.device) if gt_risk_valid_mask is not None else None
        if gt_risk_score_tensor is not None and gt_risk_score_tensor.dim() == 5:
            gt_risk_score_tensor = gt_risk_score_tensor.squeeze(2)
        if gt_risk_valid_mask_tensor is not None and gt_risk_valid_mask_tensor.dim() == 5:
            gt_risk_valid_mask_tensor = gt_risk_valid_mask_tensor.squeeze(2)
        command = self._normalize_test_field(command, squeeze_last=True, device=bev_embed.device)
        ego_his_trajs = self._normalize_test_field(ego_his_trajs, dtype=torch.float32, device=bev_embed.device)
        ego_his_trajs_mask = self._normalize_test_field(ego_his_trajs_mask, dtype=torch.float32, device=bev_embed.device)
        ego_lcf_feat = self._normalize_test_field(ego_lcf_feat, dtype=torch.float32, device=bev_embed.device)
        sdc_planning_tensor = self._normalize_test_field(sdc_planning, dtype=torch.float32, device=bev_embed.device) if sdc_planning is not None else None
        sdc_planning_mask_tensor = self._normalize_test_field(sdc_planning_mask, dtype=torch.float32, device=bev_embed.device) if sdc_planning_mask is not None else None
        goal_point_tensor = self._normalize_test_field(goal_point, dtype=torch.float32, device=bev_embed.device)
        planning_outputs = self.planning_head(
            bev_embed=bev_embed,
            risk_map=risk_outputs['final'].detach().clamp(0.0, 1.0),
            command=command,
            can_bus=can_bus,
            ego_his_trajs=ego_his_trajs,
            ego_his_trajs_mask=ego_his_trajs_mask,
            ego_lcf_feat=ego_lcf_feat,
            goal_point=goal_point_tensor,
        )

        results = []
        risk_pred = risk_outputs['final'].float().clamp(0.0, 1.0)
        for idx in range(len(img_metas)):
            result = dict(
                planning_traj=planning_outputs['final_traj'][idx].detach().cpu().numpy(),
                command=int(command[idx].detach().cpu().item()),
            )
            if gt_risk_score_tensor is not None and gt_risk_valid_mask_tensor is not None:
                target = gt_risk_score_tensor[idx].float()
                valid_mask = gt_risk_valid_mask_tensor[idx].float()
                pred = risk_pred[idx]
                abs_error = (pred - target).abs() * valid_mask
                sq_error = (pred - target).pow(2) * valid_mask
                result['risk_abs_error_sum'] = float(abs_error.sum().item())
                result['risk_sq_error_sum'] = float(sq_error.sum().item())
                result['risk_valid_count'] = float(valid_mask.sum().item())
            if sdc_planning_tensor is not None:
                result['planning_traj_gt'] = (sdc_planning_tensor[idx, 0] if sdc_planning_tensor.dim() == 4 else sdc_planning_tensor[idx]).detach().cpu().numpy()
            if sdc_planning_mask_tensor is not None:
                result['planning_traj_mask'] = (sdc_planning_mask_tensor[idx, 0] if sdc_planning_mask_tensor.dim() == 4 else sdc_planning_mask_tensor[idx]).detach().cpu().numpy()
            results.append(result)
        return bev_embed, results
