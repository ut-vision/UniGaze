import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add gazelib to path for pitchyaw_to_vector
# Try relative import first, then absolute
try:
	from gazelib.gaze.gaze_utils import pitchyaw_to_vector
except ImportError:
	# Fallback: add parent directory to path
	parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	if parent_dir not in sys.path:
		sys.path.insert(0, parent_dir)
	from gazelib.gaze.gaze_utils import pitchyaw_to_vector


class PitchYawLoss(nn.Module):
	def __init__(self, loss_type='l1', epsilon=None):
		super().__init__()
		self.loss_type = loss_type
		self.epsilon = epsilon


	def gaze_l2_loss(self, y, y_hat):
		loss = torch.abs(y - y_hat) **2   
		loss = torch.mean(loss, dim=1) 

		return loss 
		
	def gaze_l1_loss(self, y, y_hat):
		loss = torch.abs(y - y_hat) 
		loss = torch.mean(loss, dim=1) 
		return loss

	def forward(self, y, y_hat, weight=None, average=True):
		if self.loss_type == 'l1':
			loss_all =  self.gaze_l1_loss(y, y_hat)
		elif self.loss_type == 'l2':
			loss_all = self.gaze_l2_loss(y, y_hat)
		else:
			raise NotImplementedError
		
		return torch.mean(loss_all, dim=0)


class AngularGazeLoss(nn.Module):
	"""
	Geometric-aware loss function that operates on 3D gaze vectors.
	This loss respects the spherical geometry of gaze directions.
	"""
	def __init__(self, reduction='mean', eps=1e-7):
		super().__init__()
		self.reduction = reduction
		self.eps = eps

	def forward(self, pred_pitchyaw, gt_pitchyaw, weight=None):
		"""
		Args:
			pred_pitchyaw: [B, 2] predicted pitch and yaw angles
			gt_pitchyaw: [B, 2] ground truth pitch and yaw angles
			weight: [B] optional sample weights
		
		Returns:
			Angular error loss in degrees
		"""
		# Convert pitch/yaw to 3D unit vectors
		pred_vec = pitchyaw_to_vector(pred_pitchyaw)  # [B, 3]
		gt_vec = pitchyaw_to_vector(gt_pitchyaw)  # [B, 3]
		
		# Compute cosine similarity
		cos_sim = F.cosine_similarity(pred_vec, gt_vec, dim=1)  # [B]
		
		# Clamp to avoid numerical issues
		cos_sim = torch.clamp(cos_sim, -1.0 + self.eps, 1.0 - self.eps)
		
		# Compute angular error in radians, then convert to degrees
		angular_error_rad = torch.acos(cos_sim)  # [B]
		angular_error_deg = angular_error_rad * 180.0 / 3.141592653589793
		
		# Apply weights if provided
		if weight is not None:
			angular_error_deg = angular_error_deg * weight
		
		# Reduce
		if self.reduction == 'mean':
			return angular_error_deg.mean()
		elif self.reduction == 'sum':
			return angular_error_deg.sum()
		elif self.reduction == 'none':
			return angular_error_deg
		else:
			raise ValueError(f"Unknown reduction: {self.reduction}")


class CombinedGazeLoss(nn.Module):
	"""
	Combined loss that uses both angular loss and L1/L2 loss.
	This provides both geometric consistency and direct angle regression.
	"""
	def __init__(self, angular_weight=1.0, l1_weight=0.1, loss_type='l1'):
		super().__init__()
		self.angular_weight = angular_weight
		self.l1_weight = l1_weight
		self.angular_loss = AngularGazeLoss()
		self.pitchyaw_loss = PitchYawLoss(loss_type=loss_type)
	
	def forward(self, pred_pitchyaw, gt_pitchyaw, weight=None):
		angular_loss = self.angular_loss(pred_pitchyaw, gt_pitchyaw, weight)
		pitchyaw_loss = self.pitchyaw_loss(pred_pitchyaw, gt_pitchyaw, weight)
		
		total_loss = self.angular_weight * angular_loss + self.l1_weight * pitchyaw_loss
		return total_loss
		
