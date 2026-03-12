"""
Domain Adaptive Gaze Estimation
Uses adversarial training to learn domain-invariant features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
	"""
	Gradient Reversal Layer for adversarial domain adaptation.
	"""
	@staticmethod
	def forward(ctx, x, alpha):
		ctx.alpha = alpha
		return x.view_as(x)
	
	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg() * ctx.alpha
		return output, None


class GradientReversal(nn.Module):
	"""
	Gradient Reversal Layer module.
	"""
	def __init__(self, alpha=1.0):
		super().__init__()
		self.alpha = alpha
	
	def forward(self, x):
		return GradientReversalFunction.apply(x, self.alpha)


class DomainClassifier(nn.Module):
	"""
	Domain classifier for adversarial training.
	"""
	def __init__(self, input_dim, num_domains, hidden_dim=512):
		super().__init__()
		self.num_domains = num_domains
		
		self.classifier = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(hidden_dim, hidden_dim // 2),
			nn.ReLU(),
			nn.Dropout(0.5),
			nn.Linear(hidden_dim // 2, num_domains)
		)
	
	def forward(self, features):
		return self.classifier(features)


class DomainAdaptiveGaze(nn.Module):
	"""
	Domain-adaptive gaze estimation model.
	Uses adversarial training to learn domain-invariant features.
	"""
	def __init__(self, backbone, embed_dim, num_domains, head_type='linear', 
				 alpha=1.0, use_uncertainty=False):
		"""
		Args:
			backbone: Feature extraction backbone (e.g., MAE_Gaze)
			embed_dim: Dimension of features
			num_domains: Number of domains for adaptation
			head_type: Type of gaze prediction head
			alpha: Gradient reversal strength
			use_uncertainty: Whether to use uncertainty estimation
		"""
		super().__init__()
		self.backbone = backbone
		self.embed_dim = embed_dim
		self.num_domains = num_domains
		self.alpha = alpha
		
		# Freeze backbone if needed (optional)
		# for param in self.backbone.parameters():
		#     param.requires_grad = False
		
		# Gaze prediction head
		if head_type == 'linear':
			self.gaze_head = nn.Linear(embed_dim, 2)
		else:
			# Can use other head types
			from models.vit.uncertainty_gaze_head import UncertaintyGazeHead
			if use_uncertainty:
				self.gaze_head = UncertaintyGazeHead(embed_dim)
			else:
				from models.vit.multi_scale_gaze_head import MultiScaleGazeHead
				self.gaze_head = MultiScaleGazeHead(embed_dim)
		
		# Domain classifier
		self.domain_classifier = DomainClassifier(embed_dim, num_domains)
		
		# Gradient reversal layer
		self.gradient_reversal = GradientReversal(alpha)
		
		self.head_type = head_type
		self.use_uncertainty = use_uncertainty
	
	def forward(self, x, domain_labels=None, mode='train'):
		"""
		Args:
			x: [B, 3, H, W] input images
			domain_labels: [B] domain labels (for training)
			mode: 'train' or 'eval'
		
		Returns:
			dict with gaze predictions and optionally domain predictions
		"""
		# Extract features from backbone
		if hasattr(self.backbone, 'vit'):
			features = self.backbone.vit.forward_features(x)
		else:
			features = self.backbone(x)
			if isinstance(features, dict):
				features = features.get('features', features.get('pred_gaze'))
		
		output_dict = {}
		
		# Gaze prediction
		if self.head_type == 'linear':
			pred_gaze = self.gaze_head(features)
			output_dict['pred_gaze'] = pred_gaze
		elif self.use_uncertainty:
			pred_dict = self.gaze_head(features)
			output_dict.update(pred_dict)
		else:
			pred_gaze = self.gaze_head(features)
			output_dict['pred_gaze'] = pred_gaze
		
		# Domain adaptation (only during training)
		if mode == 'train' and domain_labels is not None:
			# Apply gradient reversal
			reversed_features = self.gradient_reversal(features)
			
			# Domain classification
			domain_pred = self.domain_classifier(reversed_features)
			output_dict['domain_pred'] = domain_pred
		
		return output_dict


class DomainAdaptiveLoss(nn.Module):
	"""
	Combined loss for domain-adaptive training.
	"""
	def __init__(self, gaze_loss, domain_loss_weight=1.0):
		super().__init__()
		self.gaze_loss = gaze_loss
		self.domain_loss_weight = domain_loss_weight
		self.domain_loss = nn.CrossEntropyLoss()
	
	def forward(self, pred_dict, gt_gaze, domain_labels=None):
		"""
		Args:
			pred_dict: Model output dictionary
			gt_gaze: [B, 2] ground truth gaze
			domain_labels: [B] ground truth domain labels
		"""
		# Gaze loss
		if 'pred_gaze' in pred_dict:
			pred_gaze = pred_dict['pred_gaze']
			if hasattr(self.gaze_loss, 'forward'):
				# Check if loss expects uncertainty
				if 'log_var' in pred_dict:
					gaze_loss = self.gaze_loss(pred_dict, gt_gaze)
				else:
					gaze_loss = self.gaze_loss(pred_gaze, gt_gaze)
			else:
				gaze_loss = self.gaze_loss(pred_gaze, gt_gaze)
		else:
			gaze_loss = torch.tensor(0.0, device=gt_gaze.device)
		
		# Domain loss (if available)
		if 'domain_pred' in pred_dict and domain_labels is not None:
			domain_loss = self.domain_loss(pred_dict['domain_pred'], domain_labels)
			total_loss = gaze_loss + self.domain_loss_weight * domain_loss
		else:
			total_loss = gaze_loss
			domain_loss = torch.tensor(0.0, device=gt_gaze.device)
		
		return {
			'total_loss': total_loss,
			'gaze_loss': gaze_loss,
			'domain_loss': domain_loss
		}

