
from os import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

from models.vit.mae import interpolate_pos_embed, MaskedAutoencoderViT, vit_base_patch16, vit_large_patch16, vit_huge_patch14
from models.vit.multi_scale_gaze_head import MultiScaleGazeHead
from models.vit.uncertainty_gaze_head import UncertaintyGazeHead




class MAE_Gaze(nn.Module):

	def __init__(self, model_type='vit_b_16', global_pool=False, drop_path_rate=0.1,
			  custom_pretrained_path=None, head_type='linear', return_attention=False,
			  use_uncertainty=False):
		"""
		Args:
			model_type: ViT model size ('vit_b_16', 'vit_l_16', 'vit_h_14')
			global_pool: Whether to use global pooling
			drop_path_rate: Drop path rate for regularization
			custom_pretrained_path: Path to MAE pretrained weights
			head_type: Type of prediction head ('linear', 'multi_scale', 'uncertainty')
			return_attention: Whether to return attention weights (for multi_scale head)
			use_uncertainty: Whether to use uncertainty estimation
		"""
		super().__init__()
		if model_type == "vit_b_16":
			self.vit = vit_base_patch16( global_pool=global_pool, drop_path_rate=drop_path_rate)
		elif model_type == "vit_l_16":
			self.vit = vit_large_patch16( global_pool=global_pool, drop_path_rate=drop_path_rate)
		elif model_type == "vit_h_14":
			self.vit = vit_huge_patch14( global_pool=global_pool, drop_path_rate=drop_path_rate)
		else:
			raise ValueError('model_type not supported')

		if custom_pretrained_path is not None:
			checkpoint_model = torch.load(custom_pretrained_path, map_location='cpu')['model']
			state_dict = self.vit.state_dict()
			for k in  ['head.weight', 'head.bias']:
				if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
					print(f"Removing key {k} from pretrained checkpoint")
					del checkpoint_model[k]
			

			# interpolate position embedding
			interpolate_pos_embed(self.vit, checkpoint_model)
		
			# keys_in_ckpt = checkpoint_model.keys()
			# print('Keys in ckpt: ', keys_in_ckpt)
			self.vit.load_state_dict( checkpoint_model, strict=False)
			print('Loaded custom pretrained weights from {}'.format(custom_pretrained_path))

		embed_dim = self.vit.embed_dim
		self.head_type = head_type
		self.return_attention = return_attention
		self.use_uncertainty = use_uncertainty
		
		# Initialize appropriate head
		if head_type == 'linear':
			self.gaze_head = nn.Linear(embed_dim, 2)
		elif head_type == 'multi_scale':
			self.gaze_head = MultiScaleGazeHead(
				embed_dim=embed_dim,
				return_attention=return_attention
			)
		elif head_type == 'uncertainty' or use_uncertainty:
			self.gaze_head = UncertaintyGazeHead(embed_dim=embed_dim)
		else:
			raise ValueError(f"Unknown head_type: {head_type}")


	def forward(self, input, return_features=False):
		"""
		Args:
			input: [B, 3, H, W] input images
			return_features: Whether to return intermediate features
		
		Returns:
			output_dict with 'pred_gaze' and optionally other keys
		"""
		# Get features from ViT
		features = self.vit.forward_features(input)
		
		output_dict = {}
		
		# Apply appropriate head
		if self.head_type == 'linear':
			pred_gaze = self.gaze_head(features)
			output_dict['pred_gaze'] = pred_gaze
		elif self.head_type == 'multi_scale':
			if self.return_attention:
				pred_gaze, attention_weights = self.gaze_head(features)
				output_dict['pred_gaze'] = pred_gaze
				output_dict['attention_weights'] = attention_weights
			else:
				pred_gaze = self.gaze_head(features)
				output_dict['pred_gaze'] = pred_gaze
		elif self.head_type == 'uncertainty' or self.use_uncertainty:
			pred_dict = self.gaze_head(features)
			output_dict.update(pred_dict)
		else:
			raise ValueError(f"Unknown head_type: {self.head_type}")
		
		if return_features:
			output_dict['features'] = features
		
		return output_dict
