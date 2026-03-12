"""
Gaze-Aware MAE Pre-training
Extends standard MAE with gaze-conditional masking and reconstruction
"""
import torch
import torch.nn as nn
import numpy as np
from models_mae import MaskedAutoencoderViT


class GazeConditionalMAE(MaskedAutoencoderViT):
	"""
	MAE with gaze-conditional masking and reconstruction.
	Uses gaze information to guide which patches to mask.
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# Additional decoder input for gaze conditioning
		self.gaze_embed = nn.Linear(2, self.decoder_embed_dim)  # pitch, yaw -> decoder_dim
	
	def get_eye_region_mask(self, imgs, gaze_labels, patch_size=16, img_size=224):
		"""
		Create a mask that prioritizes masking regions away from eyes.
		Eye regions are typically in the upper-middle part of face images.
		
		Args:
			imgs: [N, 3, H, W]
			gaze_labels: [N, 2] pitch and yaw (not used directly, but available)
			patch_size: Size of patches
			img_size: Size of input image
		
		Returns:
			eye_region_weights: [N, num_patches] weights for each patch
		"""
		N = imgs.shape[0]
		num_patches = (img_size // patch_size) ** 2
		patches_per_side = img_size // patch_size
		
		# Create coordinate grid for patches
		# Eye region is typically around (0.3-0.5, 0.3-0.5) in normalized coordinates
		eye_center_x, eye_center_y = 0.4, 0.4
		eye_radius = 0.15
		
		# Generate patch coordinates
		patch_coords = []
		for i in range(patches_per_side):
			for j in range(patches_per_side):
				# Normalized coordinates
				x = (j + 0.5) / patches_per_side
				y = (i + 0.5) / patches_per_side
				patch_coords.append([x, y])
		
		patch_coords = torch.tensor(patch_coords, device=imgs.device)  # [num_patches, 2]
		
		# Compute distance from eye center
		eye_center = torch.tensor([eye_center_x, eye_center_y], device=imgs.device)
		distances = torch.norm(patch_coords - eye_center.unsqueeze(0), dim=1)  # [num_patches]
		
		# Convert distance to weights (higher weight = more likely to mask)
		# Patches far from eyes get higher weight
		weights = 1.0 - torch.exp(-distances / eye_radius)
		weights = weights.unsqueeze(0).expand(N, -1)  # [N, num_patches]
		
		return weights
	
	def random_masking_with_guidance(self, guidance_weights, mask_ratio=0.75):
		"""
		Random masking with guidance from eye region weights.
		
		Args:
			guidance_weights: [N, L] weights for each patch
			mask_ratio: Target ratio of patches to mask
		
		Returns:
			mask: [N, L] binary mask (1 = mask, 0 = keep)
			ids_restore: [N, L] indices to restore original order
		"""
		N, L = guidance_weights.shape
		len_keep = int(L * (1 - mask_ratio))
		
		# Sample patches based on guidance weights
		# Higher weight = more likely to be masked
		noise = torch.rand(N, L, device=guidance_weights.device)
		# Combine guidance with random noise
		scores = guidance_weights + 0.1 * noise  # Small random component
		
		# Sort and select patches to keep (lowest scores = keep)
		ids_shuffle = torch.argsort(scores, dim=1)
		ids_restore = torch.argsort(ids_shuffle, dim=1)
		
		# Keep the first len_keep patches
		ids_keep = ids_shuffle[:, :len_keep]
		
		# Create mask (1 = mask, 0 = keep)
		mask = torch.ones(N, L, device=guidance_weights.device)
		mask[:, :len_keep] = 0
		# Unshuffle to get mask in original order
		mask = torch.gather(mask, 1, ids_restore)
		
		return mask, ids_restore
	
	def forward_encoder_gaze_aware(self, x, gaze_labels, mask_ratio=0.75):
		"""
		Gaze-aware encoding with guided masking.
		"""
		# Get eye region guidance
		guidance_weights = self.get_eye_region_mask(x, gaze_labels)
		
		# Create guided mask
		mask, ids_restore = self.random_masking_with_guidance(guidance_weights, mask_ratio)
		
		# Apply standard encoder forward
		x = self.patch_embed(x)
		
		# Add pos embed w/o cls token
		x = x + self.pos_embed[:, 1:, :]
		
		# Masking: unshuffle patches
		ids_shuffle = torch.argsort(ids_restore, dim=1)
		x = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
		
		# Keep only unmasked patches
		len_keep = int(x.shape[1] * (1 - mask_ratio))
		x = x[:, :len_keep, :]
		
		# Append cls token
		cls_token = self.cls_token + self.pos_embed[:, :1, :]
		cls_tokens = cls_token.expand(x.shape[0], -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)
		
		# Apply Transformer blocks
		for blk in self.blocks:
			x = blk(x)
		x = self.norm(x)
		
		return x, mask, ids_restore
	
	def forward_decoder_gaze_conditional(self, x, ids_restore, gaze_labels):
		"""
		Gaze-conditional decoder that uses gaze information.
		"""
		# Embed tokens
		x = self.decoder_embed(x)
		
		# Embed gaze information
		gaze_emb = self.gaze_embed(gaze_labels)  # [N, decoder_dim]
		gaze_emb = gaze_emb.unsqueeze(1)  # [N, 1, decoder_dim]
		
		# Append gaze embedding to decoder input
		x = torch.cat([x, gaze_emb], dim=1)
		
		# Add mask tokens
		mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
		x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
		x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
		x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
		
		# Add pos embed
		x = x + self.decoder_pos_embed
		
		# Apply decoder blocks
		for blk in self.decoder_blocks:
			x = blk(x)
		x = self.decoder_norm(x)
		
		# Predictor projection
		x = self.decoder_pred(x)
		
		# Remove cls token and gaze token
		x = x[:, 1:-1, :]
		
		return x
	
	def forward(self, imgs, gaze_labels=None, mask_ratio=0.75, weight=None):
		"""
		Forward pass with optional gaze conditioning.
		
		Args:
			imgs: [N, 3, H, W] input images
			gaze_labels: [N, 2] optional gaze labels for conditioning
			mask_ratio: Ratio of patches to mask
			weight: Optional pixel weights
		"""
		if gaze_labels is not None:
			# Gaze-aware forward
			latent, mask, ids_restore = self.forward_encoder_gaze_aware(imgs, gaze_labels, mask_ratio)
			pred = self.forward_decoder_gaze_conditional(latent, ids_restore, gaze_labels)
		else:
			# Standard forward
			latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
			pred = self.forward_decoder(latent, ids_restore)
		
		loss = self.forward_loss(imgs, pred, mask, weight)
		return loss, pred, mask

