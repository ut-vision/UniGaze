import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import cv2
from contextlib import nullcontext
from datetime import datetime
from rich.progress import track
from glob import glob
from torchsummary import summary
from datetime import datetime
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, ConcatDataset
import wandb
import logging
logFormatter = logging.Formatter("%(asctime)s - %(message)s")
import gc
from utils import instantiate_from_cfg
from gazelib.draw.draw_image import draw_gaze
from gazelib.utils.color_text import print_green, print_cyan, print_magenta, print_red
from gazelib.gaze.gaze_utils import angular_error
from utils.helper import recover_image, align_images, worker_init_fn, AverageMeter
from utils.util import call_model_method, get_attributes_with_prefix
import utils.misc as misc
from utils.misc import get_grad_norm_, NativeScalerWithGradNormCount


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



class SimpleTrainer(nn.Module):
	def __init__(self, config):
		nn.Module.__init__(self)
		self.config = config
		self.num_workers = config.num_workers
		self.batch_size = config.batch_size
		self.epochs = config.epochs
		self.save_epoch = config.save_epoch
		self.eval_epoch = config.eval_epoch
		self.test_per_epoch = config.test_per_epoch if 'test_per_epoch' in config else 1
		self.print_freq = config.print_freq
		self.train_iter = 0

		self.use_autocast = config.use_autocast
		self.distributed = config.distributed

		if config.distributed and config.rank != 0:
			pass
		else:
			self.ckpt_dir = os.path.join(config.output_dir, 'ckpt')
			os.makedirs(self.ckpt_dir, exist_ok=True)
			print('ckpt_dir: ', self.ckpt_dir)
			
			now = datetime.now()
			self.tensorboard_dir = os.path.join(config.output_dir, 'log'+ '_' + now.strftime("%Y%m%d_%H%M"))
			os.makedirs(self.tensorboard_dir, exist_ok=True)

			self.logging_dir = os.path.join(config.output_dir, 'logging')
			os.makedirs(self.logging_dir, exist_ok=True)

			self.image_dir = os.path.join(config.output_dir, 'image')
			os.makedirs(self.image_dir, exist_ok=True)
			
			OmegaConf.save(config, os.path.join(self.logging_dir, "project.yaml"))
			print('save config to: ', os.path.join(self.logging_dir, "project.yaml"))


		self.ckpt_resume = config.ckpt_resume
		self.metrics = {}
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.configure_datasets(config)
		self.configure_loss(config)
		self.configure_model(config)
		self.configure_hyperparams(config)
		## the self.optimizer and self.scheduler are created in the configure_hyperparams
		## so loading the state of the optimizer and scheduler should be after the configure_hyperparams
		if config.ckpt_resume is not None:
			previous_epoch = self.load_schdule(self.optimizer, self.scheduler, config.ckpt_resume)
			self.start_epoch = previous_epoch
			self.train_iter = previous_epoch * self.iteration_per_epoch

	def configure_loss(self, config):
		print(" config.loss.loss_config: ", config.loss.loss_config)
		self.gaze_loss = instantiate_from_cfg(config.loss.loss_config)

	def configure_datasets(self, config ):
		self.batch_size = config.batch_size
		self.test_batch_size = config.test_batch_size if hasattr(config, 'test_batch_size') else self.batch_size
		self.num_workers = config.num_workers
		self.use_worker_init_fn = True
		self.random_seed = config.random_seed

		data_cfg = config.data
		self.train_label_datasets = []
		self.val_loaders = {}
		self.test_loaders = {}

		for k, train_cfg_k in enumerate( data_cfg.get('train', [])):
			self.train_label_datasets.append(instantiate_from_cfg(train_cfg_k))
			print(" Initializing ", f"train_{k}", " dataset, with number of samples: ", len(self.train_label_datasets[-1]))

		if len(self.train_label_datasets) > 0:
			shuffle = config.shuffle_training if hasattr(config, 'shuffle_training') else True
			self.g = torch.Generator()
			self.g.manual_seed(self.random_seed)
			init_fn = worker_init_fn if self.use_worker_init_fn else None
			train_label_datasets_combine = ConcatDataset(self.train_label_datasets)
			if config.distributed:
				num_tasks = misc.get_world_size()
				global_rank = misc.get_rank()
				sampler_train = torch.utils.data.DistributedSampler(
						train_label_datasets_combine, num_replicas=num_tasks, rank=global_rank, shuffle=True
				)
				print("Sampler_train = %s" % str(sampler_train))
				self.train_loader = torch.utils.data.DataLoader(
					train_label_datasets_combine, sampler=sampler_train,
					batch_size=self.batch_size,
					num_workers=self.num_workers, 
					pin_memory=True,
					drop_last=True,
				)
					
			else:
				self.train_loader = torch.utils.data.DataLoader(
					train_label_datasets_combine,
					batch_size=self.batch_size,
					num_workers=self.num_workers, 
					pin_memory=True,
					drop_last=True,
					shuffle=shuffle,
					worker_init_fn=init_fn,
					generator=self.g,
				)	
			
		

		for k, val_cfg_k in enumerate( data_cfg.get('val', [])):
			val_dataset_j = instantiate_from_cfg(val_cfg_k)
			print(" Initializing val ", k, ", dataset, with number of samples: ", len(val_dataset_j))
			if config.distributed:
				num_tasks = misc.get_world_size()
				global_rank = misc.get_rank()
				sampler_val = torch.utils.data.DistributedSampler(
					val_dataset_j, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False 
				)
				print("Sampler_val = %s" % str(sampler_val))
				self.val_loaders[ f'val_dataloader_{k}'] = DataLoader( val_dataset_j, 
								sampler=sampler_val,
								batch_size=self.test_batch_size,
								num_workers=self.num_workers, 
								pin_memory=True,)		
			else:
				self.val_loaders[ f'val_dataloader_{k}'] = DataLoader( val_dataset_j,
								batch_size=self.test_batch_size,
								num_workers=self.num_workers, 
								pin_memory=True,
								shuffle=False,)

		for k, test_cfg_k in enumerate( data_cfg.get('test', [])):
			test_dataset_j = instantiate_from_cfg(test_cfg_k)
			print(" Initializing test ", k, ", dataset, with number of samples: ", len(test_dataset_j))
			if config.distributed:
				num_tasks = misc.get_world_size()
				global_rank = misc.get_rank()
				sampler_test = torch.utils.data.DistributedSampler(
					test_dataset_j, num_replicas=num_tasks, rank=global_rank, shuffle=False, drop_last=False 
				)
				print("Sampler_test = %s" % str(sampler_test))
				self.test_loaders[f'test_dataloader_{k}'] = DataLoader(test_dataset_j, 
								sampler=sampler_test,
								batch_size=self.test_batch_size,
								num_workers=self.num_workers, 
								pin_memory=True,)
			else: 
				self.test_loaders[f'test_dataloader_{k}'] = DataLoader(test_dataset_j, 
								batch_size=self.test_batch_size,
								num_workers=self.num_workers, 
								pin_memory=True,
								shuffle=False,)
			


	def configure_model(self, config):
		self.model = instantiate_from_cfg(config.model.net_config)
		self.start_epoch = 0
		self.model.to(self.device)
		if config.ckpt_resume is not None:
			self.ckpt_resume = config.ckpt_resume
			self.load_checkpoint(self.model, 'model_state', config.ckpt_resume)
			print('loaded ckpt from : ', config.ckpt_resume)

		if config.distributed:
			self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[config.gpu], find_unused_parameters=True)
			self.model_without_ddp = self.model.module
		else:
			self.model_without_ddp = self.model

		summary(self.model)
		if self.use_autocast:
			self.loss_scaler = NativeScalerWithGradNormCount()
		with torch.no_grad():
			torch.cuda.empty_cache()  # Clear cached memory
			gc.collect()
		
	def create_optimizer_and_scheduler(self, model, optimizer_config, scheduler_config):
		"""Creates an optimizer and learning rate scheduler for the given model.
		Args:
			model: The PyTorch model to optimize.
			optimizer_config: A dictionary containing optimizer parameters (e.g., 'name', 'lr', 'weight_decay').
			scheduler_config: A dictionary containing scheduler parameters (e.g., 'name', 'step_size', 'gamma').
		"""
		OPTIMIZER = getattr(optim, optimizer_config['optimizer_name'])
		optimizer_params = {k: v for k, v in optimizer_config.items() if k != 'optimizer_name'}
		optimizer = OPTIMIZER(model.parameters(), **optimizer_params)
		scheduler_name = scheduler_config['scheduler_name']
		if scheduler_name == 'StepLR':
			SCHEDULER = getattr(lr_scheduler, scheduler_name)
			scheduler_config['step_size'] = int(  float(scheduler_config['step_size']) * self.iteration_per_epoch )

			scheduler_params = {k: v for k, v in scheduler_config.items() if k != 'scheduler_name'}
			scheduler = SCHEDULER(optimizer, **scheduler_params)
			print_green(" the StepLR scheduler will update lr every {} steps".format(scheduler.step_size)) 

		elif scheduler_name == "OneCycleLR":
			SCHEDULER = getattr(lr_scheduler, scheduler_name)
			max_lr = self.config.max_lr if hasattr(self.config, 'max_lr') else optimizer_config['lr']
			epochs = self.epochs
			steps_per_epoch = self.iteration_per_epoch
			scheduler_config.update({'max_lr': max_lr, 'epochs': epochs, 'steps_per_epoch': steps_per_epoch})
			scheduler_params = {k: v for k, v in scheduler_config.items() if k != 'scheduler_name'}
			scheduler = SCHEDULER(optimizer, **scheduler_params)
		return optimizer, scheduler
	
	def configure_hyperparams(self, config ):

		self.iteration_per_epoch = len(self.train_loader)
		if hasattr(self, 'logging_dir'):
			OmegaConf.save({ 
				'iteration_per_epoch': self.iteration_per_epoch,
				}, os.path.join(self.logging_dir, "loader_params.yaml"))

		optimizer_config = config.optimizer ## refer to ""./configs/optimizers/*"
		scheduler_config = config.scheduler ## refer to ""./configs/schedulers/*"
		print(f" final_lr is just the same as base_lr: { optimizer_config['lr']}, not considering the GPU numbers"  )
		self.optimizer, self.scheduler = self.create_optimizer_and_scheduler(self.model_without_ddp, optimizer_config, scheduler_config)
		self.init_lr = self.optimizer.param_groups[0]['lr']


		self.grad_accumulation_steps = config.grad_accumulation_steps if hasattr(config, 'grad_accumulation_steps') else 1
		if hasattr(self, 'logging_dir'):
			OmegaConf.save({
				'grad_accumulation_steps': self.grad_accumulation_steps,
				}, os.path.join(self.logging_dir, "misc_params.yaml"))


		
	def train(self):
		print("\n[*] Train on {} samples (source)".format(len(self.train_loader.dataset)))
		
		if hasattr(self, 'logging_dir'):
			self.outfile = os.path.join(self.logging_dir, "train_log.log")
			rootLogger = logging.getLogger()
			fileHandler = logging.FileHandler(self.outfile)
			fileHandler.setFormatter(logFormatter)
			rootLogger.addHandler(fileHandler)
			logging.info( ' [*] Train the model. ')
		else:
			self.outfile = None

		for epoch in range(self.start_epoch, self.epochs):
			print('\nEpoch: {}/{}'.format(epoch + 1, self.epochs))
			if self.config.distributed:
				self.train_loader.sampler.set_epoch(epoch)
				print(" Set sampler epoch to ", epoch)

			self.model.train()
			self.train_one_epoch(epoch)
		
			if self.is_master():
				self.save_checkpoint(
					{'epoch': epoch + 1,
					'model_state': self.model_without_ddp.state_dict(),
					'optim_state': self.optimizer.state_dict(),
					'schedule_state': self.scheduler.state_dict()
					}, 
					add='epoch_' + str(epoch+1).zfill(2),
				)

				previous_epoch = epoch + 1 - 1
				if previous_epoch % self.config.save_epoch != 0 and previous_epoch > 0: 
					prev_checkpoint = os.path.join(self.ckpt_dir, 'epoch_' + str(previous_epoch).zfill(2) + '.pth.tar') 
					if os.path.exists(prev_checkpoint):
						os.remove(prev_checkpoint)
						print(f" deleted {prev_checkpoint}")
				
			if (epoch+1) % self.config.valid_epoch == 0 or epoch == self.epochs - 1:
				print_cyan('validate at epoch: ', epoch + 1)
				for val_key in self.val_loaders:
					self.test(epoch, eval_tag=val_key)
				
				if (epoch+1) % self.config.eval_epoch == 0 or epoch == self.epochs - 1:
					print_cyan('test at epoch: ', epoch + 1)
					for test_key in self.test_loaders:
						self.test(epoch, eval_tag=test_key)


	def is_master(self):
		if self.config.distributed:
			return self.config.rank == 0
		else:
			return True

	def train_one_epoch(self, epoch, data_loader=None):
		self.model.train()
		losses_dict = OrderedDict()
		errors_dict = OrderedDict()
		self.metrics = { }
		if data_loader is None:
			data_loader = self.train_loader
		source_iter = iter(data_loader)
		self.optimizer.zero_grad()
		for i in track(range(len(data_loader)), total=len(data_loader), description='train_iter: '):
			try: 
				entry = next(source_iter)
			except StopIteration: # restart the generator if the previous generator is exhausted.
				source_iter = iter(data_loader)
				entry = next(source_iter)
			
			vis_entry = {}
			# -------------------------------------------------
			#                    supervision
			# -------------------------------------------------
			input_var = entry['image'].float().to(self.device)
			gaze_var = entry['gaze'].float().to(self.device)
			batch_size = input_var.size(0)
			

			autocast_context = torch.cuda.amp.autocast if self.use_autocast else nullcontext
			with autocast_context():
				output_dict = self.model(input_var)
				pred_gaze = output_dict['pred_gaze']
				errors_dict['error_s_gaze'] = np.mean(angular_error(pred_gaze.cpu().data.numpy(), gaze_var.cpu().data.numpy()))
				
				# Handle different loss function signatures
				if hasattr(self.gaze_loss, 'forward'):
					# Check if loss expects uncertainty information
					if 'log_var' in output_dict or 'uncertainty' in output_dict:
						# Uncertainty-aware loss
						loss_s_gaze = self.gaze_loss(output_dict, gaze_var)
					else:
						# Standard loss
						loss_s_gaze = self.gaze_loss(pred_gaze, gaze_var)
				else:
					# Fallback to standard
					loss_s_gaze = self.gaze_loss(pred_gaze, gaze_var)
				
				losses_dict['loss_s_gaze'] = loss_s_gaze.item()
				
				# Log uncertainty if available
				if 'uncertainty' in output_dict:
					uncertainty_mean = output_dict['uncertainty'].mean().item()
					losses_dict['uncertainty_mean'] = uncertainty_mean
				self.update_meters({
					'errors_s_gaze': (errors_dict['error_s_gaze'], batch_size),
					'losses_s_gaze': (losses_dict['loss_s_gaze'], batch_size),
				})
				loss_all = 0
				loss_all += loss_s_gaze

			# backpropagation
			if self.use_autocast:
				loss_all /= self.grad_accumulation_steps
				self.loss_scaler(loss_all, self.optimizer, parameters=self.model.parameters(),
						update_grad=(self.train_iter + 1) % self.grad_accumulation_steps == 0)
				if (self.train_iter  + 1) % self.grad_accumulation_steps  == 0:
					self.optimizer.zero_grad()
				torch.cuda.synchronize()
				loss_all_log = misc.all_reduce_mean(loss_all)
			else:
				loss_all /= self.grad_accumulation_steps
				loss_all.backward()
				if (i + 1) % self.grad_accumulation_steps == 0:
					self.optimizer.step()
					self.optimizer.zero_grad()
				loss_all_log = loss_all.item()
			
			
			losses_dict['loss_all'] = loss_all_log

			self.log_save( epoch=epoch+1, iteration_per_epoch=self.iteration_per_epoch , 
						log_losses={**losses_dict, **errors_dict, **get_attributes_with_prefix(self, 'coeff_'), 
									'lr': self.optimizer.param_groups[0]['lr']},
						outfile=self.outfile, batch_size=batch_size)

			if self.is_master():
				vis_entry['input'] = (input_var, gaze_var, pred_gaze)
				self.print_entry(i, vis_entry, 'vis_train')


			self.train_iter = self.train_iter + 1
			self.scheduler.step()
		
		self.reset_meters()

	def log_save(self, epoch, iteration_per_epoch, log_losses, outfile, batch_size, optimizer=None):
		if outfile is None:# this will not print other process
			return 
		iteration = self.train_iter % iteration_per_epoch
		if iteration % self.print_freq == 0:
			for i in range(torch.cuda.device_count()):
				bytes = (torch.cuda.memory_allocated(device=i) + torch.cuda.memory_reserved(device=i))
				rank_str = f"Rank {self.config.rank}" if self.config.distributed else ""
				print_cyan(f"{rank_str} - GPU {i}: allocated approximately {bytes / 1e9} GB")
			print_cyan("====================================================================================== \n" )
			
			log = f"\n [{epoch}/{self.epochs}]: [{iteration}/{iteration_per_epoch}] \n"
			if optimizer:
				log += f"lr:{optimizer.param_groups[0]['lr']}  weight decay: {optimizer.param_groups[0]['weight_decay']} \n"
			log += f"batch_size: {batch_size} \n"
			for k,v in log_losses.items():
				log += f"{k}: {v} \n"

			logging.info(log)

			avg_log = f"\nThe average of the last {self.print_freq} iterations: \n"
			for k,v in self.metrics.items():
				avg_log += f"{k}: {v.avg} \n"
			logging.info(avg_log)
			logging.info("=====================================================================================")
			self.reset_meters()



	def test(self, epoch=None, eval_tag=None):
		if eval_tag is None: 
			for val_key, val_loader in self.val_loaders.items():
				self.test_wlabel(eval_tag=val_key, data_loader=val_loader, epoch=epoch)
			for test_key, test_loader in self.test_loaders.items():
				self.test_wlabel(eval_tag=test_key, data_loader=test_loader, epoch=epoch)
		else:
			if eval_tag in self.val_loaders:
				val_loader = self.val_loaders[eval_tag]
				self.test_wlabel(eval_tag=eval_tag, data_loader=val_loader, epoch=epoch)	
			elif eval_tag in self.test_loaders:
				test_loader = self.test_loaders[eval_tag]
				self.test_wlabel(eval_tag=eval_tag, data_loader=test_loader, epoch=epoch)

	def test_wlabel(self, eval_tag, data_loader,  epoch=None, model=None ):
		model = self.model if model is None else model
		log = ' -------------------------------------------------------------------------- \n'
		log += " [*] {} on {} samples \n".format(eval_tag, len(data_loader.dataset))
		if epoch is None:
			log += f" current self.train_iter: {self.train_iter} \n"
			log += f" total epochs: {self.epochs}  \n"
		else:
			log += f" current epoch: {epoch+1}/{self.epochs}  \n"
			log += f" current self.train_iter: {self.train_iter} \n"
		log += "\n  [#] self.optimizer's learning rate: {} \n".format(self.optimizer.param_groups[0]['lr'])
		if self.is_master():
			logging.info(log)
		model.eval()
		num_samples = len(data_loader.dataset)

		if self.config.distributed: ## get the labels and predictions from all processes
			num_samples_per_gpu =int( np.ceil( len(data_loader.dataset)/self.config.world_size )) 
			## ========================================
			gaze_all = []
			pred_gaze_all = []
			head_all = []

			subject_key_all = []

			num_iter = len(data_loader)
			for i, entry in enumerate(track(data_loader, description=' num_iter: {}  '.format(num_iter))):
				input_var = entry['image'].float().to(self.device)
				gaze_var = entry['gaze'].float().to(self.device)
				head_var = entry['head'].float().to(self.device)
				subject_key = entry['key'].to(self.device).unsqueeze(-1) ## range from [0, number_of_subjects)

				batch_size = input_var.size(0)
				gaze_all.append(gaze_var)
				head_all.append(head_var)
				subject_key_all.append(subject_key)

				## NOTE: (checked) use autocast by default, autocast() will make it much faster and the results are the same
				with torch.cuda.amp.autocast(), torch.no_grad():
					assert model.training == False, 'model should be in eval mode when testing'
					out_dict = self.model(input_var) 
					pred_gaze = out_dict['pred_gaze']
					pred_gaze_all.append(pred_gaze)

				# ========================================
			subject_key_all = torch.cat(subject_key_all, dim=0)
			gaze_all = torch.cat(gaze_all, dim=0)
			pred_gaze_all = torch.cat(pred_gaze_all, dim=0)
			head_all = torch.cat(head_all, dim=0)

			global_subject_key_all = misc.gather_tensors(subject_key_all).cpu().numpy() # gather data across all processes
			global_gaze_all = misc.gather_tensors(gaze_all).cpu().numpy() # gather data across all processes
			global_pred_gaze_all = misc.gather_tensors(pred_gaze_all).cpu().numpy()
			global_head_all = misc.gather_tensors(head_all).cpu().numpy()
			assert global_gaze_all.shape[0] == num_samples_per_gpu * self.config.world_size, 'something wrong with the multi-gpu testing data gathering'

	
			global_subject_key_all = global_subject_key_all[:num_samples].flatten().astype(int)
			global_gaze_all = global_gaze_all[:num_samples]
			global_pred_gaze_all = global_pred_gaze_all[:num_samples]
			global_head_all = global_head_all[:num_samples]



		else:
			gaze_all = np.zeros((num_samples, 2))
			pred_gaze_all = np.zeros((num_samples, 2))
			head_all = np.zeros((num_samples, 2))

			subject_key_all = np.zeros(num_samples)
			save_index = 0

			for i, entry in enumerate(track(data_loader)):
				input_var = entry['image'].float().to(self.device)
				gaze_var = entry['gaze'].float().to(self.device)
				head_var = entry['head'].float().to(self.device)
				subject_key = entry['key'] ## range from [0, number_of_subjects)

				batch_size = input_var.size(0)
				gaze_all[save_index:save_index + batch_size, :] = gaze_var.cpu().data.numpy()
				head_all[save_index:save_index + batch_size, :] = head_var.cpu().data.numpy()
				subject_key_all[save_index:save_index + batch_size] = subject_key.numpy().astype(int)

				## NOTE: (checked) use autocast by default, autocast() will make it much faster and the results are the same
				with torch.cuda.amp.autocast(), torch.no_grad():
					assert model.training == False, 'model should be in eval mode when testing'
					out_dict = self.model(input_var)
					pred_gaze = out_dict['pred_gaze']
					pred_gaze_all[save_index:save_index + batch_size, :] = pred_gaze.cpu().data.numpy()

				save_index += input_var.size(0)
				# ========================================
			if save_index != len(data_loader.dataset):
				print('the test samples save_index ', save_index, ' is not equal to the whole test set ', len(data_loader.dataset))
			global_gaze_all = gaze_all
			global_pred_gaze_all = pred_gaze_all
			global_head_all = head_all

			global_subject_key_all = subject_key_all
		if self.is_master():
			print(" global_gaze_all: ", global_gaze_all.shape)
			print(" global_pred_gaze_all: ", global_pred_gaze_all.shape)
			# -------------------------------------------------
			#                save predictions
			# -------------------------------------------------
			save_dir = os.path.join(self.logging_dir, 'prediction_'+ eval_tag)
			os.makedirs(save_dir, exist_ok=True)
			gt_gaze_path = os.path.join(save_dir, "gt_gaze.txt")
			if not os.path.exists(gt_gaze_path):
				np.savetxt(gt_gaze_path, global_gaze_all, delimiter=',')
			gt_head_path = os.path.join(save_dir, "gt_head.txt")
			if not os.path.exists(gt_head_path):
				np.savetxt(gt_head_path, global_head_all, delimiter=',')
			epoch_dir = os.path.join(save_dir, f"epoch_{epoch+1}")
			os.makedirs(epoch_dir, exist_ok=True)
			np.savetxt(os.path.join(epoch_dir, "pred_gaze.txt"), global_pred_gaze_all, delimiter=',') 

			subject_key_path = os.path.join(save_dir, "subject_key.txt")
			if not os.path.exists(subject_key_path):
				np.savetxt(subject_key_path, global_subject_key_all, delimiter=',')
			# -------------------------------------------------
			#                average error
			# -------------------------------------------------
			errors_gaze_all = angular_error(global_pred_gaze_all, global_gaze_all)
			error_gaze_avg = np.mean(errors_gaze_all)
			error_gaze_std = np.std(errors_gaze_all)
			msg_gaze = "gaze error mean: {:.3f}  -  std: {:.3f} ".format(error_gaze_avg, error_gaze_std)
			logging.info( msg_gaze )
			

		model.train()


	
	def evaluate(self):
		"""this is only called from main.py for testing a ckpt"""
		if hasattr(self, 'logging_dir'):
			# self.outfile = open(os.path.join(self.logging_dir, "eval_log.txt"), 'w')
			self.outfile = os.path.join(self.logging_dir, "eval_log.log")
			logFormatter = logging.Formatter("%(asctime)s - %(message)s")
			rootLogger = logging.getLogger()

			fileHandler = logging.FileHandler(self.outfile)
			fileHandler.setFormatter(logFormatter)
			rootLogger.addHandler(fileHandler)

			logging.info( ' [*] Evaluate the model. ')
		else:
			self.outfile = None

		epoch = 1000
		for val_key in self.val_loaders:
			self.test(epoch, eval_tag=val_key)
		for test_key in self.test_loaders:
			self.test(epoch, eval_tag=test_key)



	
	def load_schdule(self, optimizer, scheduler, ckpt_path):
		weights = torch.load(ckpt_path, map_location='cpu')
		previous_epoch = weights['epoch']
		if 'optim_state' in weights:
			optimizer.load_state_dict(weights['optim_state'])
			# optimizer.zero_grad(set_to_none=True)
			print_green('loaded optimizer state from ckpt')
		else:
			print_red('no optimizer state in ckpt, use default')
		if 'schedule_state' in weights:
			scheduler.load_state_dict(weights['schedule_state'])
			print_green('loaded schedule state from ckpt')
		else:
			print_red('no schedule state in ckpt, use default')
		return previous_epoch

	def load_checkpoint(self, model, ckpt_key, ckpt_path):
		"""
		Load the copy of a model.
		"""
		assert os.path.isfile(ckpt_path)
		weights = torch.load(ckpt_path, map_location='cpu')
		print('loaded ckpt from : ', ckpt_path)

		# If was stored using DataParallel but being read on 1 GPU
		model_state = weights[ckpt_key]
		if next(iter(model_state.keys())).startswith('module.'):
			print(' convert the DataParallel state to normal state')
			model_state = dict([(k[7:], v) for k, v in model_state.items()])

		model.load_state_dict(model_state, strict=True)
		print(f'loaded {ckpt_key}')
		del weights


	def print_entry(self, i, vis_entry, tag):
		if i % ( self.iteration_per_epoch // 10) != 0:
			return
		vis_dir = os.path.join(self.image_dir, tag)
		os.makedirs(vis_dir, exist_ok=True)
		first_key = list(vis_entry.keys())[0]
		batch_size = vis_entry[first_key][0].size(0)

		vis_entry = [ (recover_image(image_tensor, MEAN=MEAN, STD=STD), label_tensor.cpu().data.numpy(), pred_tensor.cpu().data.numpy()) 
					for entry_type, (image_tensor, label_tensor, pred_tensor) in vis_entry.items() ]

		grid = []
		num_to_vis = 40
		for b in range(min(batch_size, num_to_vis)):
			img_vis = []
			for idx, (img, label, pred) in enumerate(vis_entry):
				img_b = img[b]
				label_b = label[b]
				pred_b = pred[b]
				img_b = draw_gaze(img_b, label_b, color=(0, 255, 0))
				img_b = draw_gaze(img_b, pred_b, color=(0, 0, 255))
				img_vis.append(img_b)
			img_vis = cv2.hconcat(img_vis)
			grid.append(img_vis)
			if len(grid) == min(batch_size, num_to_vis):
				grid_img = align_images(grid, num_to_vis // 5 + 1, 5)
				cv2.imwrite(os.path.join(vis_dir, 'batch_{}_{}_{}.jpg'.format(self.train_iter, i, b)), grid_img)
				grid = []
			


	def update_meters(self, metrics_updates):
		for key, value in metrics_updates.items():
			metric_value = value[0]
			batch_size = value[1]
			if key not in self.metrics:
				self.metrics[key] = AverageMeter()
			self.metrics[key].update(metric_value, batch_size)

	def reset_meters(self, keys=None):
		if keys is None:
			keys = self.metrics.keys()
		for key in keys:
			self.metrics[key].reset()

	
	def save_checkpoint(self, state, add):
		"""
		Save a copy of the model
		"""
		filename = add + '.pth.tar'
		ckpt_path = os.path.join(self.ckpt_dir, filename)
		torch.save(state, ckpt_path)
		print('save file to: ', ckpt_path)
	


