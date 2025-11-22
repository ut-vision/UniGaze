
from os import replace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32

from .mae import interpolate_pos_embed, MaskedAutoencoderViT, vit_base_patch16, vit_large_patch16, vit_huge_patch14


from safetensors.torch import load_file as safe_load

def _read_checkpoint_any(path):
	"""
	Returns a flat dict[str, Tensor] suitable for model.load_state_dict(...)
	Supports .safetensors and .pth (optionally wrapping {'model': ...}).
	"""
	if path.endswith(".safetensors"):
		sd = safe_load(path)  # already a flat tensor dict
	else:
		ckpt = torch.load(path, map_location="cpu")
		sd = ckpt.get("model", ckpt)  # handle {'model': ...} or raw state_dict
	return sd

class MAE_Gaze(nn.Module):

	def __init__(self, model_type='vit_b_16', global_pool=False, drop_path_rate=0.1,
			  custom_pretrained_path=None):
		
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
			self.load_pretrained_mae_weights(custom_pretrained_path)


		embed_dim = self.vit.embed_dim
		self.gaze_fc = nn.Linear(embed_dim, 2)
	
	def load_pretrained_mae_weights(self, pretrained_path):
		checkpoint_model = _read_checkpoint_any(pretrained_path)
		state_dict = self.vit.state_dict()

		# Drop head if size mismatches
		for k in ['head.weight', 'head.bias']:
			if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
				print(f"Removing key {k} from pretrained checkpoint (shape mismatch {checkpoint_model[k].shape} vs {state_dict[k].shape})")
				del checkpoint_model[k]

		# Interpolate pos embedding if needed
		interpolate_pos_embed(self.vit, checkpoint_model)

		# Load (allow non-strict to tolerate minor naming diffs)
		missing, unexpected = self.vit.load_state_dict(checkpoint_model, strict=False)
		if missing or unexpected:
			print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

		print(f'Loaded custom pretrained weights from {pretrained_path}')


	def forward(self, input):
		features = self.vit.forward_features(input)

		pred_gaze = self.gaze_fc(features)
		output_dict = {}
		output_dict['pred_gaze'] = pred_gaze
		return output_dict
	