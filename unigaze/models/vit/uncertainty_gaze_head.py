"""
Uncertainty-Aware Gaze Estimation Head
Provides both gaze prediction and uncertainty quantification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyGazeHead(nn.Module):
	"""
	Uncertainty-aware gaze prediction head.
	Predicts both gaze direction and aleatoric uncertainty.
	"""
	def __init__(self, embed_dim, hidden_dim=None, dropout=0.1):
		super().__init__()
		if hidden_dim is None:
			hidden_dim = embed_dim * 2
		
		self.embed_dim = embed_dim
		self.hidden_dim = hidden_dim
		
		# Shared feature extraction
		self.shared_mlp = nn.Sequential(
			nn.Linear(embed_dim, hidden_dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.GELU(),
			nn.Dropout(dropout)
		)
		
		# Gaze prediction head
		self.gaze_head = nn.Linear(hidden_dim // 2, 2)
		
		# Uncertainty prediction head (predicts log variance for stability)
		self.uncertainty_head = nn.Linear(hidden_dim // 2, 2)
		
		# Initialize uncertainty head to predict small initial uncertainty
		nn.init.constant_(self.uncertainty_head.bias, -2.0)  # exp(-2) ≈ 0.135
	
	def forward(self, features):
		"""
		Args:
			features: [B, D] - features from backbone
		
		Returns:
			dict with keys:
				- pred_gaze: [B, 2] - predicted pitch and yaw
				- log_var: [B, 2] - log variance (aleatoric uncertainty)
				- uncertainty: [B, 2] - exp(log_var) for interpretability
		"""
		# Shared feature extraction
		shared_feat = self.shared_mlp(features)
		
		# Predict gaze
		pred_gaze = self.gaze_head(shared_feat)
		
		# Predict log variance (for numerical stability)
		log_var = self.uncertainty_head(shared_feat)
		
		# Compute uncertainty (variance)
		uncertainty = torch.exp(log_var)
		
		return {
			'pred_gaze': pred_gaze,
			'log_var': log_var,
			'uncertainty': uncertainty
		}


class UncertaintyAwareLoss(nn.Module):
	"""
	Loss function for uncertainty-aware training.
	Uses the uncertainty to weight the prediction error.
	"""
	def __init__(self, base_loss='l1', uncertainty_weight=1.0):
		super().__init__()
		self.base_loss = base_loss
		self.uncertainty_weight = uncertainty_weight
	
	def forward(self, pred_dict, gt_pitchyaw):
		"""
		Args:
			pred_dict: dict with 'pred_gaze' and 'log_var'
			gt_pitchyaw: [B, 2] ground truth
		
		Returns:
			Total loss (prediction loss + uncertainty regularization)
		"""
		pred_gaze = pred_dict['pred_gaze']
		log_var = pred_dict['log_var']
		
		# Compute base prediction error
		if self.base_loss == 'l1':
			pred_error = torch.abs(pred_gaze - gt_pitchyaw)
		elif self.base_loss == 'l2':
			pred_error = (pred_gaze - gt_pitchyaw) ** 2
		else:
			raise ValueError(f"Unknown base_loss: {self.base_loss}")
		
		# Uncertainty-weighted loss (learnable precision)
		precision = torch.exp(-log_var)  # Higher precision for lower uncertainty
		weighted_error = precision * pred_error + log_var
		
		# Sum over dimensions and average over batch
		loss = weighted_error.sum(dim=1).mean()
		
		return loss


class MonteCarloUncertaintyGazeHead(nn.Module):
	"""
	Epistemic uncertainty estimation using Monte Carlo Dropout.
	Requires multiple forward passes with dropout enabled.
	"""
	def __init__(self, base_head, num_samples=10, dropout_rate=0.1):
		super().__init__()
		self.base_head = base_head
		self.num_samples = num_samples
		self.dropout_rate = dropout_rate
	
	def forward(self, features, return_uncertainty=True):
		"""
		Args:
			features: [B, D] - features from backbone
			return_uncertainty: if True, compute epistemic uncertainty
		
		Returns:
			dict with pred_gaze and optionally epistemic_uncertainty
		"""
		if not return_uncertainty:
			return self.base_head(features)
		
		# Enable dropout for uncertainty estimation
		self.base_head.train()
		
		# Multiple forward passes
		predictions = []
		for _ in range(self.num_samples):
			pred_dict = self.base_head(features)
			predictions.append(pred_dict['pred_gaze'])
		
		# Stack predictions: [num_samples, B, 2]
		predictions = torch.stack(predictions, dim=0)
		
		# Compute mean prediction
		mean_pred = predictions.mean(dim=0)  # [B, 2]
		
		# Compute epistemic uncertainty (variance across samples)
		epistemic_var = predictions.var(dim=0)  # [B, 2]
		epistemic_std = torch.sqrt(epistemic_var + 1e-8)  # [B, 2]
		
		# Get aleatoric uncertainty from last forward pass
		last_pred_dict = self.base_head(features)
		aleatoric_uncertainty = last_pred_dict['uncertainty']
		
		return {
			'pred_gaze': mean_pred,
			'aleatoric_uncertainty': aleatoric_uncertainty,
			'epistemic_uncertainty': epistemic_std,
			'total_uncertainty': torch.sqrt(aleatoric_uncertainty + epistemic_var + 1e-8)
		}

