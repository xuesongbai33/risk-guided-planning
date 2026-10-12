import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.models import HEADS
from mmcv.utils import force_fp32

from .utils import FiLMModulation, ASPPModule


def _group_norm(channels):
    assert channels % 32 == 0, f'channels must be divisible by 32, got {channels}'
    return nn.GroupNorm(32, channels)


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size // 2)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            _group_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, dilation=dilation)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            _group_norm(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))


class DilatedContextBlock(nn.Module):
    def __init__(self, channels, dilations=(1, 2, 4)):
        super().__init__()
        self.branches = nn.ModuleList([
            ResidualBlock(channels, dilation=dilation)
            for dilation in dilations
        ])
        self.fuse = nn.Sequential(
            ConvGNAct(channels * len(dilations), channels, kernel_size=1, padding=0),
            ResidualBlock(channels),
        )

    def forward(self, x):
        return x + self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))


class SpatiotemporalOutputBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_dw = nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=channels, bias=False)
        self.spatial_pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_norm = _group_norm(channels)
        self.temporal_dw = nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=channels, bias=False)
        self.temporal_pw = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.temporal_norm = _group_norm(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.spatial_pw(self.spatial_dw(x))
        x = self.act(self.spatial_norm(x))
        x = x + residual
        residual = x
        x = self.temporal_pw(self.temporal_dw(x))
        x = self.act(self.temporal_norm(x))
        return x + residual


@HEADS.register_module()
class RiskFormerRegnitionHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 risk_h=512,
                 risk_w=512,
                 bev_h=200,
                 bev_w=200,
                 hidden_channels=160,
                 decoder_channels=128,
                 highres_channels=96,
                 output_mix_channels=32,
                 output_mix_depth=2,
                 refine_channels=64,
                 future_frames=6,
                 output_init_bias=-2.0,
                 **kwargs):
        super().__init__()
        assert not kwargs, f'Unexpected RiskFormerRegnitionHead arguments: {sorted(kwargs.keys())}'
        self.fp16_enabled = False
        self.in_channels = in_channels
        self.risk_h = risk_h
        self.risk_w = risk_w
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.future_frames = future_frames
        self.num_horizons = future_frames + 1
        self.output_mix_channels = output_mix_channels
        self.output_init_bias = output_init_bias

        coord_features = self._build_coord_features(bev_h, bev_w)
        self.register_buffer('coord_features', coord_features)

        stem_channels = in_channels + coord_features.shape[1]
        self.stem = nn.Sequential(
            ConvGNAct(stem_channels, hidden_channels),
            ResidualBlock(hidden_channels),
        )
        self.can_processor = nn.Sequential(
            nn.Linear(18, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.film = FiLMModulation(hidden_channels, hidden_channels)

        self.context_200 = DilatedContextBlock(hidden_channels, dilations=(1, 2, 4))
        self.down_100 = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels, stride=2),
            ResidualBlock(hidden_channels),
            DilatedContextBlock(hidden_channels, dilations=(1, 2, 4)),
        )
        self.down_50 = nn.Sequential(
            ConvGNAct(hidden_channels, hidden_channels, stride=2),
            ResidualBlock(hidden_channels),
            DilatedContextBlock(hidden_channels, dilations=(1, 4, 8)),
        )

        self.lat_200 = ConvGNAct(hidden_channels, hidden_channels, kernel_size=1, padding=0)
        self.lat_100 = ConvGNAct(hidden_channels, hidden_channels, kernel_size=1, padding=0)
        self.lat_50 = ConvGNAct(hidden_channels, hidden_channels, kernel_size=1, padding=0)
        self.fpn_100 = ResidualBlock(hidden_channels)
        self.fpn_200 = ResidualBlock(hidden_channels)

        self.up_256 = nn.Sequential(
            ConvGNAct(hidden_channels, decoder_channels),
            ResidualBlock(decoder_channels),
            ASPPModule(decoder_channels, decoder_channels, rates=[1, 2, 4]),
            ResidualBlock(decoder_channels),
        )
        self.up_512 = nn.Sequential(
            ConvGNAct(decoder_channels, highres_channels),
            ResidualBlock(highres_channels),
            ConvGNAct(highres_channels, refine_channels, kernel_size=1, padding=0),
            ResidualBlock(refine_channels),
        )

        self.output_seed = nn.Conv2d(decoder_channels, self.num_horizons * output_mix_channels, 1)
        self.output_mix_blocks = nn.ModuleList([
            SpatiotemporalOutputBlock(output_mix_channels) for _ in range(output_mix_depth)
        ])
        self.output_readout = nn.Sequential(
            nn.Conv3d(output_mix_channels, output_mix_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            _group_norm(output_mix_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(output_mix_channels, 1, kernel_size=1),
        )
        self.refine_detail_proj = ConvGNAct(refine_channels, refine_channels, kernel_size=3)
        self.refine_coarse_proj = ConvGNAct(1, refine_channels, kernel_size=3)
        self.output_refine = nn.Sequential(
            ResidualBlock(refine_channels),
            ConvGNAct(refine_channels, refine_channels, kernel_size=3, dilation=2),
            nn.Conv2d(refine_channels, 1, 1),
        )
        nn.init.constant_(self.output_readout[-1].bias, self.output_init_bias)
        nn.init.zeros_(self.output_refine[-1].weight)
        nn.init.zeros_(self.output_refine[-1].bias)

    @staticmethod
    def _build_coord_features(bev_h, bev_w):
        y = torch.linspace(-1.0, 1.0, bev_h, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, bev_w, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        radius = torch.sqrt(xx.square() + yy.square()) / 2.0 ** 0.5
        return torch.stack([xx, yy, radius], dim=0).unsqueeze(0)

    def _masked_mae(self, pred, target, valid_mask):
        return ((pred - target).abs() * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    @staticmethod
    def _risk_loss(pred, target, valid_mask, smooth_l1_beta=0.15, focal_beta=2.0, gamma=0.5):
        """Target-weighted Smooth L1 + over-prediction penalty.

        Two orthogonal constraints:
        1. Target-weighted Smooth L1 — global regression + peak focus (higher target, higher weight).
           smooth_l1_beta: L2→L1 transition threshold; smaller gives finer gradients in the low-error region.
           focal_beta: target weighting; larger target pixels get higher loss weight.
        2. Over-prediction penalty — penalizes background regions where pred > target.
           gamma / smooth_l1_beta scales the over-prediction gradient to match the regression gradient.
        """
        diff = pred - target
        abs_diff = diff.abs()
        smooth_l1 = torch.where(
            abs_diff < smooth_l1_beta,
            0.5 * abs_diff ** 2 / smooth_l1_beta,
            abs_diff - 0.5 * smooth_l1_beta)
        pixel_weight = 1.0 + focal_beta * target
        regression_loss = smooth_l1 * pixel_weight

        overpred = torch.clamp(pred - target, min=0.0)
        overpred_penalty = (gamma / smooth_l1_beta) * overpred.pow(2) * (1.0 - target)

        loss = regression_loss + overpred_penalty
        return (loss * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)

    def _horizon_metric(self, pred, target, valid_mask, horizon_idx):
        pred_h = pred[:, horizon_idx:horizon_idx + 1]
        target_h = target[:, horizon_idx:horizon_idx + 1]
        valid_h = valid_mask[:, horizon_idx:horizon_idx + 1]
        return self._masked_mae(pred_h, target_h, valid_h)

    def forward(self, bev_feats, can_bus):
        assert bev_feats.dim() == 3, f'bev_feats must be [B,HW,C], got {bev_feats.shape}'
        assert bev_feats.shape[1] == self.bev_h * self.bev_w
        assert bev_feats.shape[2] == self.in_channels
        b = bev_feats.shape[0]
        assert can_bus.shape == (b, 18), f'can_bus must be [B,18], got {can_bus.shape}'

        bev_feats = bev_feats.transpose(1, 2).contiguous().view(b, self.in_channels, self.bev_h, self.bev_w)
        coords = self.coord_features.to(dtype=bev_feats.dtype).expand(b, -1, -1, -1)
        x = self.stem(torch.cat([bev_feats, coords], dim=1))
        can_features = self.can_processor(can_bus.to(device=x.device, dtype=x.dtype))
        x = self.film(x, can_features)

        x200 = self.context_200(x)
        x100 = self.down_100(x200)
        x50 = self.down_50(x100)

        p50 = self.lat_50(x50)
        p100 = self.fpn_100(self.lat_100(x100) + F.interpolate(p50, size=x100.shape[-2:], mode='bilinear', align_corners=False))
        p200 = self.fpn_200(self.lat_200(x200) + F.interpolate(p100, size=x200.shape[-2:], mode='bilinear', align_corners=False))

        feature_256 = F.interpolate(p200, size=(256, 256), mode='bilinear', align_corners=False)
        feature_256 = self.up_256(feature_256)
        shared_512 = F.interpolate(feature_256, size=(self.risk_h, self.risk_w), mode='bilinear', align_corners=False)
        shared_512 = self.up_512(shared_512)

        seed = self.output_seed(feature_256)
        seed = seed.view(b, self.num_horizons, self.output_mix_channels, 256, 256).permute(0, 2, 1, 3, 4).contiguous()
        mixed = seed
        for block in self.output_mix_blocks:
            mixed = block(mixed)
        coarse_logits = self.output_readout(mixed).squeeze(1)
        coarse_logits = F.interpolate(coarse_logits, size=(self.risk_h, self.risk_w), mode='bilinear', align_corners=False)

        coarse_logits_bt = coarse_logits.reshape(b * self.num_horizons, 1, self.risk_h, self.risk_w)
        shared_512_bt = shared_512[:, None].expand(b, self.num_horizons, -1, -1, -1).reshape(b * self.num_horizons, shared_512.shape[1], self.risk_h, self.risk_w)
        refine_features = self.refine_detail_proj(shared_512_bt) + self.refine_coarse_proj(coarse_logits_bt)
        residual_logits = self.output_refine(refine_features).reshape(b, self.num_horizons, self.risk_h, self.risk_w)
        logits = coarse_logits + residual_logits
        return {
            'final': logits,
            'logits': logits,
        }

    @force_fp32(apply_to=('risk_outputs', 'risk_gt', 'gt_risk_valid_mask'))
    def loss(self, risk_outputs, risk_gt, gt_risk_valid_mask):
        assert risk_gt.dim() == 5, f'risk_gt must be [B,T,1,H,W], got {risk_gt.shape}'
        assert gt_risk_valid_mask.dim() == 5, f'valid mask must be [B,T,1,H,W], got {gt_risk_valid_mask.shape}'
        assert risk_gt.shape[1] == self.num_horizons
        assert risk_gt.shape[2] == 1
        assert risk_gt.shape[-2:] == (self.risk_h, self.risk_w)
        assert gt_risk_valid_mask.shape == risk_gt.shape
        assert risk_outputs['final'].shape == (risk_gt.shape[0], self.num_horizons, self.risk_h, self.risk_w)

        target = risk_gt.float().squeeze(2)
        valid_mask = gt_risk_valid_mask.float().squeeze(2)
        pred_raw = risk_outputs['final'].float()
        pred_clamped = pred_raw.clamp(0.0, 1.0)

        loss_risk = self._risk_loss(pred_raw, target, valid_mask)
        mae = self._masked_mae(pred_clamped, target, valid_mask)
        losses = {
            'loss_risk': loss_risk,
            'risk_final_mae': mae.detach(),
        }
        horizon_debug_indices = sorted(set([0, self.num_horizons // 2, self.num_horizons - 1]))
        for horizon_idx in horizon_debug_indices:
            losses[f'risk_final_mae_t{horizon_idx}'] = self._horizon_metric(pred_clamped, target, valid_mask, horizon_idx).detach()
        return losses
