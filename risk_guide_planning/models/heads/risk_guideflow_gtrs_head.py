import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.models import HEADS


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb)
        emb = x.float()[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Learned2DPositionalEmbedding(nn.Module):
    def __init__(self, height, width, hidden_dim):
        super().__init__()
        half_dim = hidden_dim // 2
        self.row_embed = nn.Embedding(height, half_dim)
        self.col_embed = nn.Embedding(width, hidden_dim - half_dim)
        self.height = height
        self.width = width

    def forward(self, device):
        rows = torch.arange(self.height, device=device)
        cols = torch.arange(self.width, device=device)
        row_feat = self.row_embed(rows)[:, None, :].expand(-1, self.width, -1)
        col_feat = self.col_embed(cols)[None, :, :].expand(self.height, -1, -1)
        pos = torch.cat([row_feat, col_feat], dim=-1)
        return pos.view(1, self.height * self.width, -1)


class SceneMemoryAdapter(nn.Module):
    def __init__(self, bev_channels, hidden_dim):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(bev_channels, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.position = Learned2DPositionalEmbedding(100, 100, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, bev_embed, bev_h, bev_w):
        b, _, c = bev_embed.shape
        bev_grid = bev_embed.transpose(1, 2).contiguous().view(b, c, bev_h, bev_w)
        memory = self.down(bev_grid)
        tokens = memory.flatten(2).transpose(1, 2).contiguous()
        return self.norm(tokens + self.position(tokens.device))


class MemoryTokenResampler(nn.Module):
    def __init__(self, hidden_dim, num_latents, num_heads=8, refinement_layers=1):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, hidden_dim) * (hidden_dim ** -0.5))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.refinement = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.0,
                batch_first=True,
            )
            for _ in range(refinement_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, source_tokens):
        b = source_tokens.shape[0]
        latents = self.latents.expand(b, -1, -1)
        memory = self.memory_norm(source_tokens)
        attended = self.cross_attn(self.query_norm(latents), memory, memory, need_weights=False)[0]
        latents = latents + attended
        for layer in self.refinement:
            latents = layer(latents)
        return self.output_norm(latents)


class SharedRiskMemoryEncoder(nn.Module):
    """Shared risk-map encoder: 512→256→128→64 (three stride-2 convs).

    Both generator and scorer branches share the same low-level risk features.
    This avoids computing identical convolutions twice on the same risk map.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        h = hidden_dim // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, h, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, h),
            nn.SiLU(inplace=True),
        )

    def forward(self, risk_grid):
        b, t, h, w = risk_grid.shape
        return self.encoder(risk_grid.reshape(b * t, 1, h, w))


class GeneratorRiskMemoryAdapter(nn.Module):
    def __init__(self, future_steps, hidden_dim, shared_encoder):
        super().__init__()
        self.encoder = shared_encoder
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.position = Learned2DPositionalEmbedding(64, 64, hidden_dim)
        self.time_embedding = nn.Embedding(future_steps, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, risk_map):
        risk_grid = risk_map.float()
        b, t, h, w = risk_grid.shape
        encoded = self.proj(self.encoder(risk_grid))
        tokens = encoded.flatten(2).transpose(1, 2).contiguous().reshape(b, t * 64 * 64, -1)
        pos = self.position(risk_grid.device).view(1, 1, 64 * 64, -1).expand(b, t, -1, -1).reshape(b, t * 64 * 64, -1)
        time = self.time_embedding.weight[None, :, None, :].expand(b, -1, 64 * 64, -1).reshape(b, t * 64 * 64, -1)
        return self.norm(tokens + pos + time)


class ScorerRiskMemoryAdapter(nn.Module):
    def __init__(self, future_steps, hidden_dim, shared_encoder):
        super().__init__()
        self.encoder = shared_encoder
        h = hidden_dim // 2
        self.head = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, h),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.SiLU(inplace=True),
        )
        self.position = Learned2DPositionalEmbedding(32, 32, hidden_dim)
        self.time_embedding = nn.Embedding(future_steps, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, risk_map):
        risk_grid = risk_map.float()
        b, t, h, w = risk_grid.shape
        encoded = self.head(self.encoder(risk_grid))
        tokens = encoded.flatten(2).transpose(1, 2).contiguous().reshape(b, t * 32 * 32, -1)
        pos = self.position(risk_grid.device).view(1, 1, 32 * 32, -1).expand(b, t, -1, -1).reshape(b, t * 32 * 32, -1)
        time = self.time_embedding.weight[None, :, None, :].expand(b, -1, 32 * 32, -1).reshape(b, t * 32 * 32, -1)
        return self.norm(tokens + pos + time)


class GuideFlowConditionedGenerator(nn.Module):
    """Sequential cross-attention conditioned flow-matching generator .

    Condition order: risk (safety constraints) → bev (scene layout) → dynamic cond (task intent).
    Dynamic cond (anchor, goal, command, status) is controlled via CFG.
    """

    def __init__(self, future_steps, hidden_dim, command_embed_dim):
        super().__init__()
        self.future_steps = future_steps
        T = future_steps
        self.noisy_proj = nn.Linear(4, hidden_dim)       # applied per frame: [B, T, 4] → [B, T, 256]
        self.time_proj = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Two passes through the condition pipeline, each pass = risk → bev → dynamic
        pass1 = [self._make_condition_block(hidden_dim) for _ in range(3)]
        pass2 = [self._make_condition_block(hidden_dim) for _ in range(3)]
        for name, blk in zip(['risk', 'bev', 'cond', 'risk2', 'bev2', 'cond2'],
                              pass1 + pass2):
            self.add_module(name, blk)
        self.risk_block, self.bev_block, self.cond_block = pass1
        self.risk_block_2, self.bev_block_2, self.cond_block_2 = pass2

        # Dynamic condition projections (for CFG) — global, per-timestep queries
        # attend to these via cross-attention independently.
        self.anchor_proj = nn.Linear(future_steps * 4, hidden_dim)
        self.goal_proj = nn.Linear(2, hidden_dim)
        self.goal_point_proj = nn.Linear(2, hidden_dim)
        self.command_proj = nn.Linear(command_embed_dim, hidden_dim)
        self.status_proj = nn.Linear(hidden_dim, hidden_dim)

        self.output_proj = nn.Linear(hidden_dim, 4)       # applied per frame: [B, T, 256] → [B, T, 4]
        self.query_pos = nn.Parameter(torch.zeros(1, T, hidden_dim))
        self._init_weights()

    @staticmethod
    def _make_condition_block(hidden_dim):
        return nn.ModuleDict({
            'norm_attn': nn.LayerNorm(hidden_dim),
            'norm_kv': nn.LayerNorm(hidden_dim),
            'cross_attn': nn.MultiheadAttention(hidden_dim, 8, dropout=0.0, batch_first=True),
            'norm_ffn': nn.LayerNorm(hidden_dim),
            'ffn': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(0.0),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(0.0),
            ),
        })

    @staticmethod
    def _apply_condition_block(block, h, kv):
        kv_normed = block['norm_kv'](kv)
        h = h + block['cross_attn'](block['norm_attn'](h), kv_normed, kv_normed)[0]
        h = h + block['ffn'](block['norm_ffn'](h))
        return h

    def _init_weights(self):
        for module in [self.anchor_proj, self.goal_proj, self.goal_point_proj,
                        self.command_proj, self.status_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def _build_dynamic_cond(self, anchor_traj, command_embed, status_token, goal_point):
        b = anchor_traj.shape[0]
        yaw = anchor_traj[..., 2]
        anchor_yaw = torch.cat([torch.sin(yaw), torch.cos(yaw)], dim=-1)
        return torch.stack([
            self.anchor_proj(torch.cat([
                anchor_traj[..., :2].reshape(b, -1),
                anchor_yaw], dim=-1)),
            self.goal_proj(anchor_traj[:, -1, :2]),
            self.goal_point_proj(goal_point),
            self.command_proj(command_embed),
            self.status_proj(status_token),
        ], dim=1)  # [B, 5, hidden_dim]

    def forward(self, noisy_traj, timesteps, bev_tokens, risk_tokens,
                anchor_traj, command_embed, status_token, goal_point=None):
        b = noisy_traj.shape[0]
        T = noisy_traj.shape[1]

        # Per-timestep MLP: each frame gets its own query token [B, T, 256]
        h = self.noisy_proj(noisy_traj.view(b * T, 4)).view(b, T, -1)
        h = h + self.time_proj(timesteps).unsqueeze(1)
        h = h + self.query_pos

        cond = self._build_dynamic_cond(anchor_traj, command_embed, status_token, goal_point)

        # Pass 1
        h = self._apply_condition_block(self.risk_block, h, risk_tokens)
        h = self._apply_condition_block(self.bev_block, h, bev_tokens)
        h = self._apply_condition_block(self.cond_block, h, cond)

        # Pass 2
        h = self._apply_condition_block(self.risk_block_2, h, risk_tokens)
        h = self._apply_condition_block(self.bev_block_2, h, bev_tokens)
        h = self._apply_condition_block(self.cond_block_2, h, cond)

        return self.output_proj(h).view(b, T, 4)


class CandidateTrajectoryTokenizer(nn.Module):
    """Flat-MLP trajectory tokenizer (GTRS-style).

    Trajectory geometry [x, y, sin_yaw, cos_yaw] is encoded via a plain MLP
    over all flattened steps.  Output token diversity is preserved by omitting
    output normalisation — per-trajectory norm variations carry discriminative
    signal.  Per-trajectory risk is intentionally kept out of the tokenizer so
    that the same vocab entry produces a stable token across batches; scene-level
    risk context enters through the decoder's cross-attention to scorer_memory_tokens.
    """

    def __init__(self, future_steps, hidden_dim):
        super().__init__()
        self.future_steps = future_steps
        # geometry: 4 dims (x, y, sin_yaw, cos_yaw) × future_steps
        self.traj_embed = nn.Sequential(
            nn.Linear(future_steps * 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, candidate_trajs, _candidate_risk_seq):
        b, n = candidate_trajs.shape[:2]
        yaw = candidate_trajs[..., 2]
        traj_flat = torch.cat([
            candidate_trajs[..., :2],
            torch.sin(yaw).unsqueeze(-1),
            torch.cos(yaw).unsqueeze(-1),
        ], dim=-1).view(b, n, -1)
        return self.traj_embed(traj_flat)


class IMIHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, tokens):
        return self.net(tokens).squeeze(-1)


class CrossAttentionDecoder(nn.Module):
    """Cross-attention-only decoder: each candidate attends to scene memory
    independently, without self-attention between candidates.

    Standard Pre-Norm + residual structure.  Omitting self-attention avoids
    the homogenisation that collapses cosine similarity when 2048 tokens
    attend to each other.
    """

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout),
                ),
                'norm2': nn.LayerNorm(d_model),
            }) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory):
        for layer in self.layers:
            attn_out, _ = layer['cross_attn'](layer['norm1'](tgt), memory, memory, need_weights=False)
            tgt = tgt + attn_out
            tgt = tgt + layer['ffn'](layer['norm2'](tgt))
        return tgt


class DenseTrajectoryScorer(nn.Module):
    def __init__(self, future_steps, hidden_dim, num_layers):
        super().__init__()
        self.tokenizer = CandidateTrajectoryTokenizer(future_steps, hidden_dim)
        self.decoder = CrossAttentionDecoder(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            num_layers=num_layers,
            dropout=0.0,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.imi_head = IMIHead(hidden_dim)

    def forward(self, candidate_trajs, candidate_risk_seq, scene_tokens, status_token):
        candidate_tokens = self.tokenizer(candidate_trajs, candidate_risk_seq)
        candidate_tokens = self.output_norm(self.decoder(candidate_tokens, scene_tokens))
        imi_logits = self.imi_head(candidate_tokens)
        return candidate_tokens, imi_logits


@HEADS.register_module()
class RiskGuideFlowGTRSHead(nn.Module):
    def __init__(self,
                 bev_channels=256,
                 bev_h=200,
                 bev_w=200,
                 command_dim=6,
                 command_embed_dim=32,
                 future_steps=6,
                 vocab_path=None,
                 anchor_path=None,
                 vocab_meta_path=None,
                 vocab_size=128,
                 anchor_size=32,
                 dynamic_candidates=8,
                 static_pool_size=16,
                 topk=8,
                 hidden_dim=256,
                 history_steps=4,
                 scorer_layers=3,
                 vocab_dropout=0.5,
                 point_cloud_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                 generator_loss_weight=0.2,
                 coarse_score_loss_weight=0.05,
                 imi_score_weight=1.0,
                 risk_score_weight=1.0,
                 goal_progress_score_weight=1.0,
                 dynamic_reward_score=0.0,
                 risk_mean_weight=1.0,
                 risk_max_weight=1.0,
                 risk_time_weights=(1.0, 0.85, 0.7, 0.55, 0.4, 0.25),
                 risk_blend_ratio=0.5,
                 veto_penalty=-5.0,
                 veto_penalty_far=-1.0,
                 risk_perframe_loss_weight=0.05,
                 risk_max_loss_weight=0.30,
                 imi_sigma=1.0,
                 scene_memory_latents=256,
                 generator_risk_latents=256,
                 scorer_risk_latents=128,
                 memory_refinement_layers=1,
                 num_train_anchors=8,
                 veto_thresh_near=0.70,
                 veto_thresh_far=0.90,
                 lateral_offsets=None,
                 lambda_low=0.02,
                 lambda_high=0.2,
                 **kwargs):
        super().__init__()
        self.fp16_enabled = False
        self.future_steps = future_steps
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.vocab_size = vocab_size
        self.anchor_size = anchor_size
        self.dynamic_candidates = dynamic_candidates
        self.static_pool_size = static_pool_size
        self.topk = topk
        self.hidden_dim = hidden_dim
        self.history_steps = history_steps
        self.vocab_dropout = vocab_dropout
        self.num_train_anchors = num_train_anchors
        self.point_cloud_range = tuple(float(v) for v in point_cloud_range)
        self.generator_loss_weight = generator_loss_weight
        self.coarse_score_loss_weight = coarse_score_loss_weight
        self.imi_score_weight = imi_score_weight
        self.risk_score_weight = risk_score_weight
        self.goal_progress_score_weight = goal_progress_score_weight
        self.dynamic_reward_score = dynamic_reward_score
        self.risk_mean_weight = risk_mean_weight
        self.risk_max_weight = risk_max_weight
        self.register_buffer('risk_time_weights', torch.tensor(risk_time_weights, dtype=torch.float32), persistent=False)
        self.risk_blend_ratio = risk_blend_ratio
        self.veto_penalty = float(veto_penalty)
        self.veto_penalty_far = float(veto_penalty_far)
        self.veto_thresh_near = float(veto_thresh_near)
        self.veto_thresh_far = float(veto_thresh_far)
        self.lateral_offsets = tuple(float(o) for o in lateral_offsets) if lateral_offsets else None
        self.lambda_low = float(lambda_low)
        self.lambda_high = float(lambda_high)
        self.risk_perframe_loss_weight = risk_perframe_loss_weight
        self.risk_max_loss_weight = risk_max_loss_weight
        self.imi_sigma = float(imi_sigma)
        self.risk_stat_heads = nn.ModuleDict({
            'per_frame': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, future_steps),
            ),
            'max': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, 1),
            ),
        })
        vocab = np.load(vocab_path).astype(np.float32)
        anchors = np.load(anchor_path).astype(np.float32)
        with open(vocab_meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        diff_stats = meta['residual_diff_stats']
        self.x_diff_mean = float(diff_stats['x_diff_mean'])
        self.x_diff_scale = float(diff_stats['x_diff_scale'])
        self.y_diff_mean = float(diff_stats['y_diff_mean'])
        self.y_diff_scale = float(diff_stats['y_diff_scale'])

        self.register_buffer('trajectory_vocab', torch.from_numpy(vocab), persistent=False)
        self.register_buffer('anchor_bank', torch.from_numpy(anchors), persistent=False)

        self.command_embedding = nn.Embedding(command_dim, command_embed_dim)
        self.scene_adapter = SceneMemoryAdapter(bev_channels, hidden_dim)
        shared_risk_encoder = SharedRiskMemoryEncoder(hidden_dim)
        self.generator_risk_adapter = GeneratorRiskMemoryAdapter(future_steps, hidden_dim, shared_risk_encoder)
        self.scorer_risk_adapter = ScorerRiskMemoryAdapter(future_steps, hidden_dim, shared_risk_encoder)
        self.scene_resampler = MemoryTokenResampler(hidden_dim, scene_memory_latents, refinement_layers=memory_refinement_layers)
        self.generator_risk_resampler = MemoryTokenResampler(hidden_dim, generator_risk_latents, refinement_layers=memory_refinement_layers)
        self.scorer_risk_resampler = MemoryTokenResampler(hidden_dim, scorer_risk_latents, refinement_layers=memory_refinement_layers)
        self.bev_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.risk_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.can_proj = nn.Linear(18, hidden_dim)
        self.history_proj = nn.Linear(history_steps * 2, hidden_dim)
        self.lcf_proj = nn.Linear(9, hidden_dim)
        self.command_context_proj = nn.Linear(command_embed_dim, hidden_dim)
        self.status_context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.status_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3 + command_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.flow_generator = GuideFlowConditionedGenerator(
            future_steps=future_steps,
            hidden_dim=hidden_dim,
            command_embed_dim=command_embed_dim,
        )
        self.scorer = DenseTrajectoryScorer(future_steps, hidden_dim, scorer_layers)

    @staticmethod
    def _normalize_angle(angle):
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _traj_to_diff(self, traj):
        x_diff = traj[..., 0:1].diff(dim=1, prepend=torch.zeros_like(traj[:, :1, 0:1]))
        x_diff = (x_diff - self.x_diff_mean) / max(self.x_diff_scale, 1e-6)
        y_diff = traj[..., 1:2].diff(dim=1, prepend=torch.zeros_like(traj[:, :1, 1:2]))
        y_diff = (y_diff - self.y_diff_mean) / max(self.y_diff_scale, 1e-6)
        yaw = traj[..., 2:3]
        return torch.cat([x_diff, y_diff, torch.sin(yaw), torch.cos(yaw)], dim=-1)

    def _diff_to_residual(self, diff):
        x = (diff[..., 0:1] * max(self.x_diff_scale, 1e-6) + self.x_diff_mean).cumsum(dim=1)
        y = (diff[..., 1:2] * max(self.y_diff_scale, 1e-6) + self.y_diff_mean).cumsum(dim=1)
        yaw = torch.atan2(diff[..., 2:3], diff[..., 3:4])
        return torch.cat([x, y, yaw], dim=-1)

    def _future_maps(self, maps):
        """Extract T future risk frames from [B, 1+T, H, W]."""
        return maps[:, 1:self.future_steps + 1]

    def _build_memories_and_status(self, bev_embed, risk_map_gen, risk_map_score, command, can_bus, ego_his_trajs, ego_his_trajs_mask, ego_lcf_feat):
        bev_tokens = self.scene_resampler(self.scene_adapter(bev_embed.detach(), self.bev_h, self.bev_w)) + self.bev_type_embedding
        future_risk_map_gen = self._future_maps(risk_map_gen).detach()
        future_risk_map_score = self._future_maps(risk_map_score).detach()
        risk_tokens_gen = self.generator_risk_resampler(self.generator_risk_adapter(future_risk_map_gen)) + self.risk_type_embedding
        risk_tokens_scorer = self.scorer_risk_resampler(self.scorer_risk_adapter(future_risk_map_score)) + self.risk_type_embedding

        history_feat = self.history_proj((ego_his_trajs * ego_his_trajs_mask).reshape(bev_embed.shape[0], -1))
        can_feat = self.can_proj(can_bus)
        lcf_feat = self.lcf_proj(ego_lcf_feat)
        command_feat = self.command_embedding(command.long())
        status_token = self.status_proj(torch.cat([history_feat, can_feat, lcf_feat, command_feat], dim=-1))
        command_context = self.command_context_proj(command_feat).unsqueeze(1)
        status_context = self.status_context_proj(status_token).unsqueeze(1)
        scorer_memory_tokens = torch.cat([bev_tokens, risk_tokens_scorer, command_context, status_context], dim=1)
        return {
            'bev_tokens': bev_tokens,
            'risk_tokens_gen': risk_tokens_gen,
            'scorer_memory': scorer_memory_tokens,
            'status_token': status_token,
            'command_feat': command_feat,
        }

    def _distance(self, trajs, refs):
        xy_error = (trajs[:, None, :, :2] - refs[:, :, :, :2]) ** 2
        yaw_error = self._normalize_angle(trajs[:, None, :, 2] - refs[:, :, :, 2]).abs()
        return xy_error.sum(dim=(2, 3)) + 0.25 * yaw_error.sum(dim=2)

    def _nearest_anchor(self, target_traj, command):
        anchors = self.anchor_bank
        distance = self._distance(target_traj, anchors.unsqueeze(0))
        nearest_idx = distance.argmin(dim=1)
        batch_index = torch.arange(target_traj.shape[0], device=target_traj.device)
        return anchors[nearest_idx], nearest_idx

    @staticmethod
    def _fps_select(points, scores, k):
        """Farthest-point sampling: select K points with high scores and spatial diversity."""
        b, n = scores.shape[:2]
        best_idx = scores.argmax(dim=1, keepdim=True)
        selected = [best_idx]
        dist_to_selected = torch.full((b, n), float('inf'), device=points.device)

        for _ in range(k - 1):
            latest = selected[-1].unsqueeze(-1).expand(-1, -1, 2)
            latest_pt = points.gather(1, latest).squeeze(1)
            dist = (points - latest_pt.unsqueeze(1)).norm(dim=-1)
            dist_to_selected = torch.min(dist_to_selected, dist)
            composite = dist_to_selected * torch.sigmoid(scores)
            composite = composite.scatter(1, torch.cat(selected, dim=1), -float('inf'))
            next_idx = composite.argmax(dim=1, keepdim=True)
            selected.append(next_idx)

        return torch.cat(selected, dim=1)  # [B, K]

    def _make_generator_candidates(self, bev_tokens, risk_tokens_gen, status_token, command_feat,
                                    anchor_candidates, goal_point, risk_map):
        b, k, _, _ = anchor_candidates.shape
        flat_anchors = anchor_candidates.reshape(b * k, self.future_steps, 3)
        # Expand scene tokens
        rep_bev = bev_tokens.repeat_interleave(k, dim=0)
        rep_risk = risk_tokens_gen.repeat_interleave(k, dim=0)
        rep_status = status_token.repeat_interleave(k, dim=0)
        rep_command = command_feat.repeat_interleave(k, dim=0)
        rep_goal = goal_point.repeat_interleave(k, dim=0)

        # Rectified flow: zt = (1-t)*noise + t*target, target_velocity = target - noise
        # Inference: integrate from pure noise to data via RK4
        x = torch.randn(b * k, self.future_steps, 4, device=bev_tokens.device)
        steps = 5
        dt = 1.0 / steps

        def _v_pred(x_, t_):
            return self.flow_generator(
                x_, t_, rep_bev, rep_risk, flat_anchors,
                rep_command, rep_status, goal_point=rep_goal)

        def _evaluate(x_eval, t_eval):
            return self._risk_guide_velocity(
                _v_pred(x_eval, t_eval), x_eval, flat_anchors, risk_map)

        for k_step in range(steps):
            t_k = torch.full((b * k,), k_step * dt, device=x.device, dtype=torch.float32)
            t_mid = t_k + dt / 2
            t_next = t_k + dt

            k1 = _evaluate(x, t_k)
            k2 = _evaluate(x + dt * k1 / 2, t_mid)
            k3 = _evaluate(x + dt * k2 / 2, t_mid)
            k4 = _evaluate(x + dt * k3, t_next)
            x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        residual = self._diff_to_residual(x)
        traj = flat_anchors + residual
        traj_yaw = self._normalize_angle(traj[..., 2])
        traj = torch.cat([traj[..., :2], traj_yaw.unsqueeze(-1)], dim=-1)

        # Bounded risk refinement on the final trajectory: step downhill along the
        # time-aligned risk gradient, clipping per-waypoint displacement to 0.2 m.
        risk_grad = self._sample_risk_gradient(
            traj.view(b, k, self.future_steps, 3), risk_map
        ).view(b * k, self.future_steps, 2)  # [B*K, T, 2] physical-space raw gradient
        delta = -5.0 * risk_grad
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        delta = delta * torch.minimum(
            torch.ones_like(delta_norm), 0.2 / delta_norm)
        traj = traj.clone()
        traj[..., :2] = traj[..., :2] + delta
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        traj[..., 0] = traj[..., 0].clamp(x_min, x_max)
        traj[..., 1] = traj[..., 1].clamp(y_min, y_max)

        return traj.view(b, k, self.future_steps, 3)

    def _risk_guide_velocity(self, v_pred, noise, flat_anchors, risk_map):
        """Subtract velocity component parallel to risk gradient direction.

        v_pred is the flow-matching velocity in normalised-diff space (d(z_norm)/dt).
        Risk gradient is in physical space.  The two are related by:
            v_phys[i] = scale * Σ_{j≤i} v_pred[j]
        Correction is computed in physical space and inverse-transformed back.
        """
        residual = self._diff_to_residual(noise.detach())
        traj = flat_anchors + residual
        traj_yaw = self._normalize_angle(traj[..., 2])
        traj = torch.cat([traj[..., :2], traj_yaw.unsqueeze(-1)], dim=-1)
        n_total = traj.shape[0]
        b_map = risk_map.shape[0]
        k_per = n_total // b_map
        traj_4d = traj.view(b_map, k_per, self.future_steps, 3)

        risk_seq = self._sample_sequence_field(
            traj_4d, risk_map).view(n_total, self.future_steps)
        max_risk = risk_seq.max(dim=-1).values

        risk_weight = torch.sigmoid((max_risk - 0.5) / 0.1667)
        lambda_risk = (self.lambda_low + (self.lambda_high - self.lambda_low) * risk_weight).clamp(0, self.lambda_high)

        risk_grad = self._sample_risk_gradient(
            traj_4d, risk_map).view(n_total, self.future_steps, 2)  # [N, T, 2], physical

        v_full = v_pred.reshape(n_total, self.future_steps, 4)
        v_xy = v_full[..., :2]  # [N, T, 2], normalised-diff space

        # diff → physical: v_phys[i] = scale * cumsum_{j≤i} v_pred[j]
        sx = max(self.x_diff_scale, 1e-6)
        sy = max(self.y_diff_scale, 1e-6)
        v_phys_x = sx * v_xy[..., 0].cumsum(dim=1)
        v_phys_y = sy * v_xy[..., 1].cumsum(dim=1)

        # Per-frame correction in physical space
        grad_norm = risk_grad.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        grad_dir = risk_grad / grad_norm
        toward_risk = torch.clamp(
            v_phys_x * grad_dir[..., 0] + v_phys_y * grad_dir[..., 1], min=0)  # [N, T]

        correction_phys_x = lambda_risk.unsqueeze(-1) * toward_risk * grad_dir[..., 0]  # [N, T]
        correction_phys_y = lambda_risk.unsqueeze(-1) * toward_risk * grad_dir[..., 1]  # [N, T]

        # physical → diff: inverse of cumsum is discrete derivative
        prep_x = torch.zeros_like(correction_phys_x[:, :1])
        prep_y = torch.zeros_like(correction_phys_y[:, :1])
        correction_diff_x = torch.diff(correction_phys_x, dim=1, prepend=prep_x) / sx
        correction_diff_y = torch.diff(correction_phys_y, dim=1, prepend=prep_y) / sy

        v_xy_corrected = torch.stack([
            v_xy[..., 0] - correction_diff_x,
            v_xy[..., 1] - correction_diff_y,
        ], dim=-1)
        return torch.cat([v_xy_corrected, v_full[..., 2:]], dim=-1).reshape_as(v_pred)

    def _sample_risk_gradient(self, trajs, risk_map):
        """Central-difference spatial gradient of risk at trajectory points.

        Uses symmetric ±x/±y perturbations for higher accuracy (O(eps^2)) and
        unbiased direction/magnitude estimation.
        """
        eps = 1.0
        b, n, t_traj, _ = trajs.shape

        dxp = trajs.clone()
        dxp[..., 0] += eps
        dxm = trajs.clone()
        dxm[..., 0] -= eps
        dyp = trajs.clone()
        dyp[..., 1] += eps
        dym = trajs.clone()
        dym[..., 1] -= eps

        # Single grid_sample for all 5 positions (center, +x, -x, +y, -y)
        future_maps = self._future_maps(risk_map)
        h, w = future_maps.shape[-2], future_maps.shape[-1]
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range

        all_pts = torch.cat([trajs, dxp, dxm, dyp, dym], dim=1)  # [B, 5*N, T, 3]
        gx = (all_pts[..., 0].float() - x_min) / max(x_max - x_min, 1e-6) * 2 - 1
        gy = (all_pts[..., 1].float() - y_min) / max(y_max - y_min, 1e-6) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).permute(0, 2, 1, 3).reshape(b * t_traj, n * 5, 1, 2)
        feat = future_maps.reshape(b * t_traj, 1, h, w)
        sampled = F.grid_sample(
            feat, grid, mode='bilinear', padding_mode='border', align_corners=True,
        ).view(b, t_traj, n * 5).permute(0, 2, 1)  # [B, N*5, T]

        risk = sampled[:, :n]          # center
        risk_dxp = sampled[:, n:2 * n]  # +x
        risk_dxm = sampled[:, 2 * n:3 * n]  # -x
        risk_dyp = sampled[:, 3 * n:4 * n]  # +y
        risk_dym = sampled[:, 4 * n:5 * n]  # -y

        grad_x = (risk_dxp - risk_dxm) / (2 * eps)
        grad_y = (risk_dyp - risk_dym) / (2 * eps)
        return torch.stack([grad_x, grad_y], dim=-1)  # [B, N, T, 2]

    def _sample_sequence_field(self, trajs, field_maps, lateral_offsets=None):
        field_maps = self._future_maps(field_maps)
        b, n, t, _ = trajs.shape
        h = field_maps.shape[-2]
        w = field_maps.shape[-1]
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range

        x = trajs[..., 0].float()  # [B, N, T]
        y = trajs[..., 1].float()

        if lateral_offsets is not None:
            # Lateral footprint: lateral direction from each waypoint yaw (-sin(yaw), cos(yaw)).
            yaw = trajs[..., 2]  # [B, N, T]
            lat_x = -torch.sin(yaw)
            lat_y = torch.cos(yaw)
            off = torch.tensor(lateral_offsets, device=trajs.device, dtype=x.dtype).view(1, 1, 1, -1)
            x = x.unsqueeze(-1) + lat_x.unsqueeze(-1) * off  # [B, N, T, O]
            y = y.unsqueeze(-1) + lat_y.unsqueeze(-1) * off
        else:
            x = x.unsqueeze(-1)  # [B, N, T, 1]
            y = y.unsqueeze(-1)

        no = x.shape[-1]
        gx = (x - x_min) / max(x_max - x_min, 1e-6) * 2 - 1
        gy = (y - y_min) / max(y_max - y_min, 1e-6) * 2 - 1
        # Permute to [B,T,N,O,2] so waypoint k samples risk frame k.
        grid = torch.stack([gx, gy], dim=-1).permute(0, 2, 1, 3, 4).reshape(b * t, n * no, 1, 2)
        feat = field_maps.reshape(b * t, 1, h, w)
        sampled = F.grid_sample(feat, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        sampled = sampled.view(b, t, n, no).permute(0, 2, 1, 3).contiguous()  # [B, N, T, no]
        return sampled.max(dim=-1).values  # [B, N, T]

    def _select_static_vocab_for_training(self, command):
        vocab = self.trajectory_vocab  # [M, 6, 3]
        b = command.shape[0]
        if not self.training or self.vocab_dropout <= 0:
            return vocab.unsqueeze(0).expand(b, -1, -1, -1)
        keep = max(1, int(vocab.shape[0] * (1.0 - self.vocab_dropout)))
        rand = torch.rand(vocab.shape[0], device=vocab.device)
        indices = rand.topk(k=keep, dim=0, largest=False).indices
        return vocab[indices].unsqueeze(0).expand(b, -1, -1, -1)

    def _topk_indices(self, scores, k):
        return torch.topk(scores, k=min(k, scores.shape[1]), dim=1).indices

    def _imi_bce_labels(self, candidates, target, mask):
        """Per-candidate absolute similarity: exp(-dist/sigma). Independent of other candidates."""
        diff = (candidates - target[:, None]) * mask[:, None]
        valid_steps = mask[..., 0].sum(dim=-1, keepdim=True).clamp(min=1.0)
        distance = (diff[..., :2].pow(2).sum(dim=-1).sum(dim=-1) +
                    0.25 * diff[..., 2].abs().sum(dim=-1)) / valid_steps
        return torch.exp(-distance / max(self.imi_sigma, 1e-6))  # [B, N]

    def _risk_rule_composite(self, risk_seq):
        w = self.risk_time_weights  # [T] higher weight for near frames
        mean = (risk_seq * w).sum(dim=-1) / w.sum()
        max_risk = risk_seq.max(dim=-1).values
        return -(self.risk_mean_weight * mean +
                 self.risk_max_weight * max_risk)

    def _segment_risk(self, trajs, risk_map, lateral_offsets=None):
        """Per-segment midpoint risk, returning [B, N, T].

        T segments: ego→wpt[0] (sampled on current frame) and wpt[k-1]→wpt[k]
        (sampled on future frame k).  Each segment is time-aligned with the
        corresponding risk frame so that time-weighted veto can be applied.

        Args:
            trajs:    [B, N, T, 3]  candidate trajectories (LiDAR frame)
            risk_map: [B, 1+T, H, W]  risk field (t0, t1, …, tT)
            lateral_offsets: optional, lateral footprint offsets (inference only)
        Returns:
            seg_risk: [B, N, T]  risk value at each segment midpoint
        """
        b, n, t, _ = trajs.shape
        future = self._future_maps(risk_map)  # [B, T, H, W]
        full_risk = torch.cat([risk_map[:, 0:1], future], dim=1)  # [B, T+1, H, W]
        h, w = full_risk.shape[-2], full_risk.shape[-1]
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range

        ego = torch.zeros(b, n, 1, 2, device=trajs.device)
        wpt0 = trajs[:, :, :1, :2]  # [B, N, 1, 2]
        mid_ego_to_w0 = (ego + wpt0) / 2  # [B, N, 1, 2]

        mid_w_to_w = (trajs[:, :, :-1, :2] + trajs[:, :, 1:, :2]) / 2  # [B, N, T-1, 2]

        all_mids = torch.cat([mid_ego_to_w0, mid_w_to_w], dim=2)  # [B, N, T, 2]

        if lateral_offsets is not None:
            # Segment direction: ego→w0 and w_{k-1}→w_k; lateral is its unit perpendicular
            dirs = torch.cat([wpt0, trajs[:, :, 1:, :2] - trajs[:, :, :-1, :2]], dim=2)  # [B,N,T,2]
            dn = dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            lat = torch.stack([-dirs[..., 1], dirs[..., 0]], dim=-1) / dn  # [B,N,T,2]
            off = torch.tensor(lateral_offsets, device=trajs.device, dtype=trajs.dtype).view(1, 1, 1, -1)
            mids = all_mids.unsqueeze(-2) + lat.unsqueeze(-2) * off.unsqueeze(-1)  # [B,N,T,O,2]
        else:
            mids = all_mids.unsqueeze(-2)  # [B,N,T,1,2]

        no = mids.shape[-2]
        gx = (mids[..., 0] - x_min) / max(x_max - x_min, 1e-6) * 2 - 1  # [B,N,T,O]
        gy = (mids[..., 1] - y_min) / max(y_max - y_min, 1e-6) * 2 - 1

        risk_vals = []
        for k in range(t):
            grid_k = torch.stack([gx[:, :, k], gy[:, :, k]], dim=-1).reshape(b * n, no, 1, 2)
            feat_k = full_risk[:, k, :, :].unsqueeze(1).expand(-1, n, -1, -1).reshape(b * n, 1, h, w)
            r_k = F.grid_sample(feat_k, grid_k, mode='bilinear',
                                padding_mode='border', align_corners=True)
            risk_vals.append(r_k.view(b, n, no).max(dim=-1).values)  # [B, N]
        return torch.stack(risk_vals, dim=-1)  # [B, N, T]

    def _sample_ego_risk(self, risk_map):
        """Sample the risk at ego's current LiDAR-frame position (origin) on t0.

        Returns:
            ego_risk: [B]  risk value at the origin on the current frame
        """
        b = risk_map.shape[0]
        h, w = risk_map.shape[-2], risk_map.shape[-1]
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        gx = (0.0 - x_min) / max(x_max - x_min, 1e-6) * 2 - 1
        gy = (0.0 - y_min) / max(y_max - y_min, 1e-6) * 2 - 1
        grid = torch.tensor([[[[gx, gy]]]], device=risk_map.device,
                            dtype=risk_map.dtype).expand(b, 1, 1, 2)  # [B, 1, 1, 2]
        feat = risk_map[:, 0:1, :, :]  # [B, 1, H, W]
        sampled = F.grid_sample(feat, grid, mode='bilinear',
                                padding_mode='border', align_corners=True)
        return sampled.view(b)  # [B]

    def _time_weighted_veto(self, risk_seq, trajs=None, risk_map=None, lateral_offsets=None):
        """Time-weighted soft veto: strict near, lenient far, worst single point only.

        The planner only executes the nearest one or two waypoints, so near
        waypoints should veto more strongly than far ones.  Take the single
        most-severely-vetoed waypoint (rather than summing all of them) so the
        veto magnitude stays bounded.  Threshold and penalty interpolate over
        waypoint time.

        Args:
            risk_seq: [B, N, T] per-waypoint risk values
            trajs:    [B, N, T, 3] optional, for segment-midpoint veto
            risk_map: [B, 1+T, H, W] optional, for ego-risk lower bound + segment veto
            lateral_offsets: optional, lateral footprint offsets (inference only)
        Returns:
            veto_score: [B, N] non-positive single-point max veto penalty (0 = no veto)
        """
        b, n, t = risk_seq.shape
        device = risk_seq.device
        time_idx = torch.arange(t, device=device, dtype=torch.float32)
        frac = time_idx / max(t - 1, 1)

        # Threshold and penalty interpolate over waypoint time: strict near, lenient far
        thresh = self.veto_thresh_near + \
            (self.veto_thresh_far - self.veto_thresh_near) * frac  # [T]
        penalty = self.veto_penalty + \
            (self.veto_penalty_far - self.veto_penalty) * frac  # [T], <= 0

        # Lower-bound the near threshold by ego's current risk
        if risk_map is not None:
            ego_risk = self._sample_ego_risk(risk_map).unsqueeze(-1)  # [B, 1]
            thresh = torch.maximum(thresh.unsqueeze(0), ego_risk + 0.1)  # [B, T]
        else:
            thresh = thresh.unsqueeze(0)  # [1, T]

        # Merge waypoint risk and segment-midpoint risk into per-step max
        per_step = risk_seq  # [B, N, T]
        if trajs is not None and risk_map is not None:
            seg_risk = self._segment_risk(trajs, risk_map, lateral_offsets)  # [B, N, T]
            per_step = torch.maximum(per_step, seg_risk)

        exceed = (per_step > thresh.unsqueeze(1)).float()  # [B, N, T]
        severity = exceed * penalty.view(1, 1, t)  # [B, N, T], 0 or negative
        veto_score = severity.min(dim=-1).values  # [B, N] most severe point (most negative)
        return veto_score

    def _risk_learned_composite(self, decoded):
        per_frame = torch.sigmoid(self.risk_stat_heads['per_frame'](decoded))
        mean_pred = per_frame.mean(dim=-1)
        max_pred = torch.sigmoid(self.risk_stat_heads['max'](decoded)).squeeze(-1)
        return -(self.risk_mean_weight * mean_pred +
                 self.risk_max_weight * max_pred)

    def _risk_composite(self, risk_seq, decoded, trajs=None, risk_map=None, apply_veto=True, lateral_offsets=None):
        rule = self._risk_rule_composite(risk_seq)
        learned = self._risk_learned_composite(decoded)
        composite = self.risk_blend_ratio * rule + (1.0 - self.risk_blend_ratio) * learned
        if apply_veto:
            composite = composite + self._time_weighted_veto(risk_seq, trajs, risk_map, lateral_offsets)
        return composite

    def _goal_rule_score(self, candidate_trajs, goal_point):
        """Score by forward progress toward the goal (ego at LiDAR origin).

        progress = displacement · goal_dir penalizes both stopping (progress≈0)
        and lateral deviation (smaller dot product), with no extra alignment term.
        """
        displacement = candidate_trajs[:, :, -1, :2]  # [B, N, 2]
        goal_dir = goal_point / (goal_point.norm(dim=-1, keepdim=True) + 1e-8)  # [B, 2]
        progress = (displacement * goal_dir[:, None, :]).sum(dim=-1)  # [B, N]
        return torch.sigmoid(progress / 10.0)

    def _goal_composite(self, candidate_trajs, goal_point):
        return self._goal_rule_score(candidate_trajs, goal_point)

    def _aggregate_scores(self, imi_logits, risk_composite, goal_composite):
        return (
            self.imi_score_weight * torch.sigmoid(imi_logits) +
            self.risk_score_weight * risk_composite +
            self.goal_progress_score_weight * goal_composite
        )

    def forward(self,
                bev_embed,
                risk_map,
                command,
                can_bus,
                ego_his_trajs,
                ego_his_trajs_mask,
                ego_lcf_feat,
                sdc_planning=None,
                gt_risk_map=None,
                goal_point=None):
        mem = self._build_memories_and_status(
            bev_embed=bev_embed,
            risk_map_gen=risk_map,
            risk_map_score=risk_map,
            command=command,
            can_bus=can_bus,
            ego_his_trajs=ego_his_trajs,
            ego_his_trajs_mask=ego_his_trajs_mask,
            ego_lcf_feat=ego_lcf_feat,
        )
        bev_tokens = mem['bev_tokens']
        risk_tokens_gen = mem['risk_tokens_gen']
        scorer_memory_tokens = mem['scorer_memory']
        status_token = mem['status_token']
        command_feat = mem['command_feat']

        static_vocab = self._select_static_vocab_for_training(command)
        anchor_candidates = self.anchor_bank.unsqueeze(0).expand(
            command.shape[0], -1, -1, -1)  # [M,6,3] → [B,M,6,3]
        if self.training:
            dynamic_candidates = None
            scorer_candidates = static_vocab
            candidate_risk_seq = self._sample_sequence_field(scorer_candidates, risk_map.detach()).clamp_(0.0, 1.0)
            decoded, coarse_imi_logits = self.scorer(scorer_candidates, candidate_risk_seq, scorer_memory_tokens, status_token)
            coarse_risk_composite = self._risk_composite(candidate_risk_seq, decoded)
            coarse_goal_composite = self._goal_composite(scorer_candidates, goal_point)
            coarse_candidate_scores = self._aggregate_scores(coarse_imi_logits, coarse_risk_composite, coarse_goal_composite)
        else:
            # Score anchors → pick high-score anchors (no veto)
            with torch.no_grad():
                a_risk = self._sample_sequence_field(
                    anchor_candidates, risk_map.detach(), lateral_offsets=self.lateral_offsets).clamp_(0.0, 1.0)
                a_dec, a_imi = self.scorer(anchor_candidates, a_risk, scorer_memory_tokens, status_token)
                a_risk_c = self._risk_composite(a_risk, a_dec, apply_veto=False)
                a_goal_c = self._goal_composite(anchor_candidates, goal_point)
                a_scores = self._aggregate_scores(a_imi, a_risk_c, a_goal_c)
                # Anchor mix: half high-score (feasibility), half FPS (spatial diversity).
                k_high = self.dynamic_candidates // 2
                k_fps = self.dynamic_candidates - k_high
                high_idx = a_scores.topk(k=k_high, dim=1).indices  # [B, k_high]
                fps_mask = torch.zeros_like(a_scores, dtype=torch.bool).scatter(1, high_idx, True)
                fps_scores = a_scores.masked_fill(fps_mask, float('-inf'))
                fps_idx = self._fps_select(
                    anchor_candidates[:, :, -1, :2], fps_scores, k_fps)  # [B, k_fps]
                selected_idx = torch.cat([high_idx, fps_idx], dim=1)  # [B, dynamic_candidates]
                selected_anchors = torch.gather(
                    anchor_candidates, 1,
                    selected_idx[:, :, None, None].expand(-1, -1, self.future_steps, 3))

            dynamic_candidates = self._make_generator_candidates(
                bev_tokens, risk_tokens_gen, status_token, command_feat,
                selected_anchors, goal_point, risk_map.detach(),
            )
            # Score static vocab → pick top-(topk - dynamic_candidates)
            s_risk = self._sample_sequence_field(
                static_vocab, risk_map.detach(), lateral_offsets=self.lateral_offsets).clamp_(0.0, 1.0)
            s_dec, s_imi = self.scorer(static_vocab, s_risk, scorer_memory_tokens, status_token)
            s_risk_c = self._risk_composite(
                s_risk, s_dec, static_vocab, risk_map.detach(), lateral_offsets=self.lateral_offsets)
            s_goal_c = self._goal_composite(static_vocab, goal_point)
            s_scores = self._aggregate_scores(s_imi, s_risk_c, s_goal_c)
            topk_s = self._topk_indices(s_scores, self.static_pool_size)

            # Score dynamic candidates
            d_risk = self._sample_sequence_field(
                dynamic_candidates, risk_map.detach(), lateral_offsets=self.lateral_offsets).clamp_(0.0, 1.0)
            d_dec, d_imi = self.scorer(dynamic_candidates, d_risk, scorer_memory_tokens, status_token)

            # ---- Feasibility prefilter: mark physically infeasible dynamic candidates ----
            _, n_dyn, _, _ = dynamic_candidates.shape
            dyn_steps = (dynamic_candidates[:, :, 1:, :2] - dynamic_candidates[:, :, :-1, :2]).norm(dim=-1)  # [B,K,5]
            dyn_bad = (dyn_steps.max(dim=-1).values > 15.0) | \
                      (dynamic_candidates[:, :, 0, :2].norm(dim=-1) > 15.0) | \
                      (dynamic_candidates[:, :, :, 0].abs().max(dim=-1).values > (
                          self.point_cloud_range[3] - self.point_cloud_range[0]) / 2) | \
                      (dynamic_candidates[:, :, :, 1].abs().max(dim=-1).values > (
                          self.point_cloud_range[4] - self.point_cloud_range[1]) / 2)

            # Combine into scoring pool
            g_traj = topk_s[:, :, None, None].expand(-1, -1, self.future_steps, 3)
            g_tok = topk_s[:, :, None].expand(-1, -1, s_dec.shape[-1])
            static_trajs = torch.gather(static_vocab, 1, g_traj)
            static_dec = torch.gather(s_dec, 1, g_tok)
            static_imi = torch.gather(s_imi, 1, topk_s)
            static_risk_c = torch.gather(s_risk_c, 1, topk_s)
            static_goal_c = torch.gather(s_goal_c, 1, topk_s)
            static_risk_seq = torch.gather(s_risk, 1, topk_s[:, :, None].expand(-1, -1, s_risk.shape[-1]))
            dyn_risk_c = self._risk_composite(d_risk, d_dec, dynamic_candidates, risk_map.detach(),
                                              lateral_offsets=self.lateral_offsets)
            dyn_goal_c = self._goal_composite(dynamic_candidates, goal_point)

            scorer_candidates = torch.cat([static_trajs, dynamic_candidates], dim=1)
            decoded = torch.cat([static_dec, d_dec], dim=1)
            coarse_imi_logits = torch.cat([static_imi, d_imi], dim=1)
            coarse_risk_composite = torch.cat([static_risk_c, dyn_risk_c], dim=1)
            coarse_goal_composite = torch.cat([static_goal_c, dyn_goal_c], dim=1)
            candidate_risk_seq = torch.cat([static_risk_seq, d_risk], dim=1)
            n_dyn = dynamic_candidates.shape[1]

            coarse_candidate_scores = self._aggregate_scores(coarse_imi_logits, coarse_risk_composite, coarse_goal_composite)
            # Set infeasible dynamic candidate scores to -inf
            if n_dyn > 0:
                dyn_start = coarse_candidate_scores.shape[1] - n_dyn
                coarse_candidate_scores[:, dyn_start:] = coarse_candidate_scores[:, dyn_start:].masked_fill(
                    dyn_bad, float('-inf'))
        batch_idx = torch.arange(scorer_candidates.shape[0], device=scorer_candidates.device)

        if self.training:
            # Training: hierarchical imi — 8192→topk second-stage scoring.
            topk_indices = self._topk_indices(coarse_candidate_scores, self.topk)
            gather_traj = topk_indices[:, :, None, None].expand(-1, -1, self.future_steps, 3)
            topk_trajs = torch.gather(scorer_candidates, 1, gather_traj)
            refined_trajs = topk_trajs
            refined_risk_seq = self._sample_sequence_field(
                refined_trajs, risk_map.detach(), lateral_offsets=self.lateral_offsets).clamp_(0.0, 1.0)
            refined_decoded, final_imi_logits = self.scorer(refined_trajs, refined_risk_seq, scorer_memory_tokens, status_token)
            topk_scores = torch.gather(coarse_candidate_scores, 1, topk_indices)
            final_risk_composite = self._risk_composite(
                refined_risk_seq, refined_decoded, refined_trajs, risk_map, lateral_offsets=self.lateral_offsets)
            final_goal_composite = self._goal_composite(refined_trajs, goal_point)
            final_candidate_scores = self._aggregate_scores(final_imi_logits, final_risk_composite, final_goal_composite)
            best_idx = final_candidate_scores.argmax(dim=1)
            final_traj = refined_trajs[batch_idx, best_idx]
        else:
            # Inference: score the combined pool directly and argmax, adding a
            # fixed base score to the dynamic candidates.
            candidate_scores = coarse_candidate_scores
            if self.dynamic_reward_score != 0.0 and n_dyn > 0:
                dyn_start = candidate_scores.shape[1] - n_dyn
                candidate_scores = candidate_scores.clone()
                candidate_scores[:, dyn_start:] = candidate_scores[:, dyn_start:] + self.dynamic_reward_score
            best_idx = candidate_scores.argmax(dim=1)
            final_traj = scorer_candidates[batch_idx, best_idx]
            topk_indices = None
            topk_trajs = scorer_candidates
            topk_scores = candidate_scores
            refined_trajs = scorer_candidates
            refined_risk_seq = candidate_risk_seq
            final_imi_logits = coarse_imi_logits
            final_risk_composite = coarse_risk_composite
            final_candidate_scores = candidate_scores

        outputs = {
            'candidate_trajs': scorer_candidates,
            'dynamic_candidates': dynamic_candidates,
            'coarse_candidate_scores': coarse_candidate_scores,
            'coarse_candidate_imi_logits': coarse_imi_logits,
            'candidate_risk_seq': candidate_risk_seq,
            'coarse_risk_composite': coarse_risk_composite,
            'topk_indices': topk_indices,
            'topk_trajs': topk_trajs,
            'topk_scores': topk_scores,
            'refined_trajs': refined_trajs,
            'refined_risk_seq': refined_risk_seq,
            'final_candidate_imi_logits': final_imi_logits,
            'final_risk_composite': final_risk_composite,
            'final_candidate_scores': final_candidate_scores,
            'final_scores': final_candidate_scores,
            'final_traj': final_traj,
        }

        outputs['decoded'] = decoded
        outputs['goal_point'] = goal_point
        if gt_risk_map is not None and self.training:
            # gt_risk_map from dataset is [B, 1, T, H, W] → squeeze to [B, T, H, W]
            gt = gt_risk_map.detach()
            if gt.dim() == 5:
                gt = gt.squeeze(2)
            outputs['gt_risk_seq'] = self._sample_sequence_field(
                scorer_candidates, gt
            )

        if sdc_planning is not None:
            target = sdc_planning[:, 0] if sdc_planning.dim() == 4 else sdc_planning  # [B, 6, 3]
            b = target.shape[0]
            K = self.num_train_anchors
            K_near = K // 2
            K_fps = K - K_near
            anchors = self.anchor_bank  # [M, 6, 3]
            anchors_b = anchors.unsqueeze(0).expand(b, -1, -1, -1)  # [B, M, 6, 3]
            distance = self._distance(target, anchors_b)  # [B, M]

            # nearest anchors (first K_near)
            _, near_idx = distance.topk(k=K_near, dim=1, largest=False)  # [B, K_near]

            # FPS anchors (scorer → FPS picks K_fps)
            with torch.no_grad():
                a_risk = self._sample_sequence_field(anchors_b, risk_map.detach()).clamp_(0.0, 1.0)
                a_dec, a_imi = self.scorer(anchors_b, a_risk, scorer_memory_tokens, status_token)
                a_rc = self._risk_composite(a_risk, a_dec)
                a_gc = self._goal_composite(anchors_b, goal_point)
                a_scores = self._aggregate_scores(a_imi, a_rc, a_gc)
                fps_idx = self._fps_select(anchors_b[:, :, -1, :2], a_scores, K_fps)  # [B, K_fps]

            # merge: [B, K, 6, 3]
            sel_near = anchors[near_idx]                           # [B, K_near, 6, 3]
            sel_fps = anchors[fps_idx]                             # [B, K_fps, 6, 3]
            train_anchors = torch.cat([sel_near, sel_fps], dim=1)   # [B, K, 6, 3]

            # expand to [B*K, 6, 3]
            flat_anchors = train_anchors.reshape(b * K, self.future_steps, 3)
            target_rep = target.unsqueeze(1).expand(-1, K, -1, -1).reshape(b * K, self.future_steps, 3)

            target_residual = target_rep.clone()
            target_residual[..., :2] -= flat_anchors[..., :2]
            target_residual[..., 2] = self._normalize_angle(
                target_residual[..., 2] - flat_anchors[..., 2])
            target_diff = self._traj_to_diff(target_residual)  # [B*K, 6, 4]

            # independent noise + timestep per anchor
            z0 = torch.randn_like(target_diff)
            t = torch.rand(b * K, device=target_diff.device).clamp_(1e-3, 1.0 - 1e-3)
            zt = t[:, None, None] * target_diff + (1.0 - t[:, None, None]) * z0
            target_velocity = target_diff - z0

            # repeat BEV/risk/command/status/goal K times
            rep_bev = bev_tokens.repeat_interleave(K, dim=0)
            rep_risk = risk_tokens_gen.repeat_interleave(K, dim=0)
            rep_cmd = command_feat.repeat_interleave(K, dim=0)
            rep_st = status_token.repeat_interleave(K, dim=0)
            rep_goal = goal_point.repeat_interleave(K, dim=0)

            pred_velocity = self.flow_generator(
                zt, t, rep_bev, rep_risk, flat_anchors,
                rep_cmd, rep_st, goal_point=rep_goal)
            outputs['generator_loss'] = F.mse_loss(pred_velocity, target_velocity)
        return outputs

    def loss(self,
             planning_outputs,
             sdc_planning,
             sdc_planning_mask):
        target = sdc_planning[:, 0] if sdc_planning.dim() == 4 else sdc_planning
        mask_xy = sdc_planning_mask[:, 0] if sdc_planning_mask.dim() == 4 else sdc_planning_mask
        target = target.to(device=planning_outputs['final_traj'].device, dtype=planning_outputs['final_traj'].dtype)
        mask_xy = mask_xy.to(device=target.device, dtype=target.dtype)
        mask = torch.cat([mask_xy, mask_xy[..., :1]], dim=-1)

        candidate_trajs = planning_outputs['candidate_trajs']
        coarse_imi_sim = self._imi_bce_labels(candidate_trajs, target, mask)
        coarse_imi_logits = planning_outputs['coarse_candidate_imi_logits']
        coarse_score_loss = F.binary_cross_entropy_with_logits(coarse_imi_logits, coarse_imi_sim)

        topk_trajs = planning_outputs['topk_trajs']
        topk_imi_sim = self._imi_bce_labels(topk_trajs, target, mask)
        topk_imi_logits = planning_outputs['final_candidate_imi_logits']
        topk_score_loss = F.binary_cross_entropy_with_logits(topk_imi_logits, topk_imi_sim)

        losses = {
            'loss_imi_coarse': coarse_score_loss * self.coarse_score_loss_weight,
            'loss_imi_topk': topk_score_loss * self.coarse_score_loss_weight,
        }
        if 'generator_loss' in planning_outputs:
            losses['loss_planning_generator'] = planning_outputs['generator_loss'] * self.generator_loss_weight

        decoded = planning_outputs.get('decoded')
        gt_risk_seq = planning_outputs.get('gt_risk_seq')
        if decoded is not None and gt_risk_seq is not None:
            per_frame_pred = self.risk_stat_heads['per_frame'](decoded)
            losses['loss_risk_perframe'] = self.risk_perframe_loss_weight * \
                F.smooth_l1_loss(per_frame_pred, gt_risk_seq)
            max_risk_label = gt_risk_seq.max(dim=-1).values
            losses['loss_risk_max'] = self.risk_max_loss_weight * \
                F.smooth_l1_loss(self.risk_stat_heads['max'](decoded).squeeze(-1), max_risk_label)
        return losses
