"""
Multi-Scale Feature Fusion Head with Attention Mechanisms
Provides better feature utilization and interpretable attention maps
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleGazeHead(nn.Module):
	"""
	Multi-scale attention-based gaze prediction head.
	Uses attention pooling to focus on relevant spatial regions.
	"""
	def __init__(self, embed_dim, num_heads=8, num_layers=2, dropout=0.1, return_attention=False):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.return_attention = return_attention
		
		# Multi-head attention for spatial feature aggregation
		self.attention_pool = nn.MultiheadAttention(
			embed_dim=embed_dim,
			num_heads=num_heads,
			dropout=dropout,
			batch_first=False
		)
		
		# Learnable query token for attention pooling
		self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))
		
		# Multi-layer MLP for final prediction
		mlp_hidden_dim = embed_dim * 2
		self.mlp = nn.Sequential(
			nn.Linear(embed_dim, mlp_hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(mlp_hidden_dim // 2, 2)  # pitch, yaw
		)
		
		# Layer normalization
		self.norm = nn.LayerNorm(embed_dim)
	
	def forward(self, features):
		"""
		Args:
			features: [B, N, D] or [B, D] - features from ViT
				If [B, D], assumes global pooling already done
				If [B, N, D], uses attention pooling
		
		Returns:
			pred_gaze: [B, 2] - predicted pitch and yaw
			attention_weights: [B, 1, N] - attention weights (if return_attention=True)
		"""
		batch_size = features.shape[0]
		
		# Handle both global pooled and spatial features
		if len(features.shape) == 2:
			# Already pooled, just use MLP
			features = self.norm(features)
			pred_gaze = self.mlp(features)
			if self.return_attention:
				return pred_gaze, None
			return pred_gaze
		
		# Spatial features: [B, N, D]
		seq_len = features.shape[1]
		
		# Expand query token for batch
		query = self.query_token.expand(1, batch_size, -1)  # [1, B, D]
		
		# Transpose for attention: [N, B, D]
		features_t = features.transpose(0, 1)
		
		# Apply attention pooling
		pooled, attention_weights = self.attention_pool(
			query, features_t, features_t
		)  # pooled: [1, B, D], attention_weights: [B, 1, N]
		
		# Squeeze sequence dimension
		pooled = pooled.squeeze(0)  # [B, D]
		
		# Normalize and predict
		pooled = self.norm(pooled)
		pred_gaze = self.mlp(pooled)
		
		if self.return_attention:
			return pred_gaze, attention_weights
		return pred_gaze


class FeaturePyramidGazeHead(nn.Module):
	"""
	Feature Pyramid Network style head that fuses multi-scale features.
	"""
	def __init__(self, embed_dim, num_scales=3, dropout=0.1):
		super().__init__()
		self.embed_dim = embed_dim
		self.num_scales = num_scales
		
		# Scale-specific projections
		self.scale_projections = nn.ModuleList([
			nn.Sequential(
				nn.Linear(embed_dim, embed_dim),
				nn.GELU(),
				nn.Dropout(dropout)
			) for _ in range(num_scales)
		])
		
		# Feature fusion
		self.fusion = nn.Sequential(
			nn.Linear(embed_dim * num_scales, embed_dim * 2),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(embed_dim * 2, embed_dim),
			nn.GELU(),
			nn.Dropout(dropout)
		)
		
		# Final prediction head
		self.pred_head = nn.Linear(embed_dim, 2)
	
	def forward(self, features_list):
		"""
		Args:
			features_list: List of [B, D] features at different scales
		
		Returns:
			pred_gaze: [B, 2]
		"""
		# Project each scale
		projected = []
		for i, feat in enumerate(features_list):
			proj_feat = self.scale_projections[i](feat)
			projected.append(proj_feat)
		
		# Concatenate and fuse
		fused = torch.cat(projected, dim=1)  # [B, D * num_scales]
		fused = self.fusion(fused)  # [B, D]
		
		# Predict
		pred_gaze = self.pred_head(fused)
		return pred_gaze

