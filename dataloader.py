import numpy as np
import random
import os
import time
import torch
import torch.nn as nn
import torch.utils.data as data
import json
from PIL import Image
from torchvision import transforms
from scipy.io import loadmat
import cv2
from glob import glob

# mapping between joints and LSP IDs
id2joint = {0:'ankle',
			1: 'knee',
			2: 'hip',
			3: 'hip',
			4: 'knee',
			5: 'ankle',
			6: 'wrist',
			7: 'elbow',
			8: 'shoulder',
			9: 'shoulder',
			10: 'elbow',
			11: 'wrist'
			}

joint2id = {'ankle': [0, 5],
			'knee': [1, 4],
			'hip': [2, 3],
			'wrist': [6, 11],
			'elbow': [7, 10],
			'shoulder': [8,9]
			}

joint_encoding = {'ankle': 0,
				  'knee': 1,
				  'hip': 2,
				  'wrist': 3,
				  'elbow': 4,
				  'shoulder': 5
				}

id_connectivity = {0: [1],
				1: [0, 2],
				2: [1, 8],
				3: [4, 9],
				4: [3, 5],
				5: [4],
				6: [7],
				7: [6, 8],
				8: [2, 7],
				9: [3, 10],
				10: [9, 11],
				11: [10]	
				}

class LSP_joint_generator(data.Dataset):
	""" Dataloader for processing LSP joint data
	"""
	def __init__(self, img_dir, anno_dir, split='train', joint_size=36):
		""" Initialization function

		Inputs:
			img_dir: Directory storing image files
			anno_dir: Directory storing annotation files
			mode: Specify data split, 'train', 'val', or 'test'
			joint_size: Spatial size of the joint patches
		"""

		self.img_dir = img_dir
		self.split = split
		self.joint_size = joint_size
		self.rescaler = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		anno_file = loadmat(os.path.join(anno_dir, 
					'joints_scaled_cropped.mat'))['joints_scaled_cropped']
		self.excluded_data = json.load(open(os.path.join(
									anno_dir, 'excluded_human_data.json')))
		self.cluster_assignment = json.load(open(os.path.join(
									anno_dir, 'cluster_assignment.json')))
		self.cluster_category= np.load(os.path.join(anno_dir, 
								'cluster_assignment_categorized.npy'), allow_pickle=True).item()
		self.image_size = json.load(open(os.path.join(
									anno_dir, 'image_size.json')))
		self.split_info = json.load(open(os.path.join(
									anno_dir, 'split_info_latest.json')))			
		# self.recorded_acc = json.load(open(os.path.join(anno_dir, 'pred_confidence.json')))		
		self.recorded_acc = json.load(open('pred_confidence_relation_vgg.json'))		

		self.init_data(anno_file)

	def init_data(self, anno_file):
		# get the number of clustering categories
		self.cluster_cat = 0
		for joint in self.cluster_category:
			self.cluster_cat += len(self.cluster_category[joint])
		self.cluster2joint = torch.zeros(6, self.cluster_cat)
		residual = 0
		for idx, joint in enumerate(joint_encoding):
			self.cluster2joint[idx, residual:residual+len(self.cluster_category[joint])] =1
			residual += len(self.cluster_category[joint])

		# using pre-sampled training/val splits
		if self.split != 'test':
			id_pool = [int(cur)-1 for cur in self.split_info[self.split]]
		else:
			# id_pool = [int(cur)-1 for cur in self.excluded_data]
			id_pool = [int(cur)-1 for cur in self.split_info[self.split]]


		# iterate through all selected images and their corresponding joints
		self.annotation = []
		for img_id in id_pool:
			# for human experiments
			if self.split == 'test' and str(img_id+1) not in self.recorded_acc:
				continue

			h, w = self.image_size[str(img_id+1)]

			# only considering 6 pairs of joints
			for joint_id in range(12):
				# for special-same or human (need to +1)
				if not str(joint_id) in self.recorded_acc[str(img_id+1)]['low']:
					continue
				x, y = anno_file[0, joint_id, img_id], anno_file[1, joint_id, img_id]
				if x > w or y > h:
					continue
				self.annotation.append({'image': img_id+1,
										'joint': joint_id, 
										'location':[int(y)-1, int(x)-1] # note that the order is reversed
										})

	def crop_joint(self, image, joint_loc):
		""" Function for cropping region centered at
			a joint location.

			Inputs:
			image: a tensor containing the whole image.
			joint_loc: location of the joint.

			Return:
			A tensor containing the cropped joint
		"""

		# determine the boundary of cropped joint
		_, height, width = image.shape
		y1 = max(0, joint_loc[0]-int(self.joint_size/2))
		y2 = min(joint_loc[0]+int(self.joint_size/2), height)
		x1 = max(0, joint_loc[1]-int(self.joint_size/2))
		x2 = min(joint_loc[1]+int(self.joint_size/2), width)
		# crop and rescale
		crop_joint = image[:, y1:y2, x1:x2]
		crop_joint = self.rescaler(crop_joint)

		return crop_joint

	def __getitem__(self, index):
		data = self.annotation[index]
		img_id = data['image']
		joint_id = data['joint']
		joint_loc = data['location']
		data_id = str(img_id) + '_' + str(joint_id)

		# loading image
		img =  Image.open(os.path.join(self.img_dir, 
						'im'+str(img_id).zfill(4))+'_scaled_cropped.jpg'
						).convert('RGB')
		img = transforms.ToTensor()(img)

		# obtaining the cropped region centered at joint
		joint_img = self.crop_joint(img, joint_loc)
		# joint_img = transforms.functional.rotate(joint_img, 0)

		# category of joint
		joint_label = torch.zeros(6,)		
		joint_label[joint_encoding[id2joint[joint_id]]] = 1
		
		# standard categorization of joint relationship
		relation_mask = torch.zeros(self.cluster_cat,)
		cluster_data = self.cluster_assignment[str(img_id)][str(joint_id)]
		cluster_info = tuple([joint_id] + list(cluster_data.values()))
		cluster_id = self.cluster_category[id2joint[joint_id]][cluster_info]
		base_residual = 0
		for joint in joint_encoding:
			if joint != id2joint[joint_id]:
				base_residual += len(self.cluster_category[joint])
			else:
				break
		relation_mask[base_residual+cluster_id] = 1


		return joint_img, joint_label, relation_mask, data_id

	def __len__(self,):
		return len(self.annotation)

class LSP_pair_generator(data.Dataset):
	""" Dataloader for processing LSP joint pairs.
	"""
	def __init__(self, img_dir, anno_dir, split='train', joint_size=36, 
				connected_only=False, ablation=False, rotate=0):
		""" Initialization function

		Inputs:
			img_dir: Directory storing image files
			anno_dir: Directory storing annotation files
			mode: Specify data split, 'train', 'val', or 'test'
			joint_size: Spatial size of the joint patches
			connected_only: Only consider connected pairs or not
			ablation: Testing with an ablation experiment on a specific test set
			rotate: Degree of rotation (cue-only) for ablation study
		"""


		self.img_dir = img_dir
		self.split = split
		self.joint_size = joint_size
		self.connected_only = connected_only
		self.ablation = ablation
		self.rotate = rotate
		self.rescaler = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		anno_file = loadmat(os.path.join(anno_dir, 
					'joints_scaled_cropped.mat'))['joints_scaled_cropped']
		self.excluded_data = json.load(open(os.path.join(
									anno_dir, 'excluded_human_data.json')))
		self.cluster_assignment = json.load(open(os.path.join(
									anno_dir, 'cluster_assignment.json')))
		self.cluster_category= np.load(os.path.join(anno_dir, 
								'cluster_assignment_categorized.npy'), allow_pickle=True).item()
		self.image_size = json.load(open(os.path.join(
									anno_dir, 'image_size.json')))
		self.split_info = json.load(open(os.path.join(
									anno_dir, 'split_info_latest.json')))		
		if self.ablation:
			# self.recorded_acc = json.load(open(os.path.join(anno_dir, 'pred_confidence_36.json')))		
			self.recorded_acc = json.load(open('pred_confidence_relation_vgg.json'))		

		self.init_data(anno_file)

	def init_data(self, anno_file):
		# get the number of clustering categories
		self.cluster_cat = 0
		for joint in self.cluster_category:
			self.cluster_cat += len(self.cluster_category[joint])
		self.cluster2joint = torch.zeros(6, self.cluster_cat)
		residual = 0
		for idx, joint in enumerate(joint_encoding):
			self.cluster2joint[idx, residual:residual+len(self.cluster_category[joint])] =1
			residual += len(self.cluster_category[joint])

		# using pre-sampled training/val splits
		if self.split != 'test':
			id_pool = [int(cur)-1 for cur in self.split_info[self.split]]
		else:
			# id_pool = [int(cur)-1 for cur in self.excluded_data]
			id_pool = [int(cur)-1 for cur in self.split_info[self.split]]

		# iterate through all selected images and their corresponding joints
		self.annotation = []
		for img_id in id_pool:
			# # for human experiment
			# if self.ablation and str(img_id+1) not in self.recorded_acc:
			# 	continue

			h, w = self.image_size[str(img_id+1)]

			# use randomly sampled joint pairs for training 
			if self.split == 'train':
				tmp = {'image': img_id+1, 'joint': {}}
				for joint_id in range(12):
					x, y = anno_file[0, joint_id, img_id], anno_file[1, joint_id, img_id]
					if x > w or y > h:
						continue
					tmp['joint'][joint_id] = [int(y)-1, int(x)-1]

				self.annotation.append(tmp)
			# consider all possible pairs for evaluation
			else:
				if not self.ablation:
					target_range = range(12)
					cue_range = range(12)
				else:
					target_range = self.recorded_acc[str(img_id+1)]['low']
					target_range = [int(cur) for cur in target_range] # -1 for human data		
					cue_range = self.recorded_acc[str(img_id+1)]['high']
					cue_range = [int(cur) for cur in cue_range]

				for target_id in target_range:
					target_x, target_y = anno_file[0, target_id, img_id], anno_file[1, target_id, img_id]
					if target_x > w or target_y > h:
						continue

					for cue_id in cue_range:
						# only consider pairs between different joints
						if id2joint[cue_id] == id2joint[target_id]:
							continue

						# remove invalid data
						cue_x, cue_y = anno_file[0, cue_id, img_id], anno_file[1, cue_id, img_id]
						if cue_x > w or cue_y > h:
							continue

						# only consider connected pair
						connected = 1 if cue_id in id_connectivity[target_id] else 0
							
						tmp = {'image': img_id+1, 
								'pair': [target_id, cue_id],
								'location': [[int(target_y)-1, int(target_x)-1],
											[int(cue_y)-1, int(cue_x)-1]],
								'connectivity': connected

								}
						self.annotation.append(tmp)
		print(len(self.annotation))
		assert 0

	def crop_joint(self, image, joint_loc, cue_size=None):
		""" Function for cropping region centered at
			a joint location.

			Inputs:
			image: a tensor containing the whole image.
			joint_loc: location of the joint.
			cue_size: testing with a specific cue size

			Return:
			A tensor containing the cropped joint
		"""

		# determine the boundary of cropped joint
		_, height, width = image.shape
		if cue_size is None:
			y1 = max(0, joint_loc[0]-int(self.joint_size/2))
			y2 = min(joint_loc[0]+int(self.joint_size/2), height)
			x1 = max(0, joint_loc[1]-int(self.joint_size/2))
			x2 = min(joint_loc[1]+int(self.joint_size/2), width)
		else:
			y1 = max(0, joint_loc[0]-int(cue_size/2))
			y2 = min(joint_loc[0]+int(cue_size/2), height)
			x1 = max(0, joint_loc[1]-int(cue_size/2))
			x2 = min(joint_loc[1]+int(cue_size/2), width)

		# crop and rescale
		crop_joint = image[:, y1:y2, x1:x2]
		crop_joint = self.rescaler(crop_joint)

		return crop_joint

	def __getitem__(self, index):
		if self.split == 'train':
			data = self.annotation[index]
			img_id = data['image']

			# loading image
			img =  Image.open(os.path.join(self.img_dir, 
							'im'+str(img_id).zfill(4))+'_scaled_cropped.jpg'
							).convert('RGB')
			img = transforms.ToTensor()(img)

			if not self.connected_only:
				# sample two joints for training
				target_joint, cue_joint = np.random.choice(
											list(data['joint'].keys()), 2, replace=False)
			else:
				flag = False # 15 joints are invalid, and need to be considered as special cases
				while not flag:
					target_joint = np.random.choice(list(data['joint'].keys()), 1)
					connected_joints = id_connectivity[target_joint]
					for connected_joint in connected_joints:
						if connected_joint in data['joint']:
							flag = True
							break
				cue_joint = np.random.choice(connected_joints, 1)

			
			cropped_joint = []
			joint_label = []
			relation_label = []
			for idx, joint_id in enumerate([target_joint, cue_joint]):
				joint_loc = data['joint'][joint_id]

				# obtaining the cropped region centered at joint
				joint_img = self.crop_joint(img, joint_loc)
				cropped_joint.append(joint_img)

				# category of joint
				cur_label = torch.zeros(6,)		
				cur_label[joint_encoding[id2joint[joint_id]]] = 1
				joint_label.append(cur_label)
			
				# standard categorization of joint relationship
				relation_mask = torch.zeros(self.cluster_cat,)
				cluster_data = self.cluster_assignment[str(img_id)][str(joint_id)]
				cluster_info = tuple([joint_id] + list(cluster_data.values()))
				cluster_id = self.cluster_category[id2joint[joint_id]][cluster_info]
				base_residual = 0
				for joint in joint_encoding:
					if joint != id2joint[joint_id]:
						base_residual += len(self.cluster_category[joint])
					else:
						break
				relation_mask[base_residual+cluster_id] = 1
				relation_label.append(relation_mask)

			# compute the relative location of joints (from cue to target)
			relative_loc = [data['joint'][cue_joint][0]-data['joint'][target_joint][0],
							data['joint'][cue_joint][1]-data['joint'][target_joint][1]]
			relative_loc = torch.FloatTensor(relative_loc)

			return cropped_joint[0], cropped_joint[1], relation_label[0], relation_label[1], joint_label[0], joint_label[1], relative_loc

		# iterative through all possible pairs for evaluations
		else:
			data = self.annotation[index]
			img_id = data['image']

			# loading image
			img =  Image.open(os.path.join(self.img_dir, 
							'im'+str(img_id).zfill(4))+'_scaled_cropped.jpg'
							).convert('RGB')
			img = transforms.ToTensor()(img)

			# get pre-extracted joints
			joint_id = data['pair']
			joint_loc = data['location']

			cropped_joint = []
			joint_label = []
			relation_label = []
			for idx in range(2):
				# obtain joint image
				# joint_img = self.crop_joint(img, joint_loc[idx])

				if idx == 1:
					joint_img = self.crop_joint(img, joint_loc[idx], 60) # ablation with specific cue size
					if self.rotate >0:
						joint_img = transforms.functional.rotate(joint_img, self.rotate)
				else:
					joint_img = self.crop_joint(img, joint_loc[idx])
					# if self.rotate >0:
					# 	joint_img = transforms.functional.rotate(joint_img, self.rotate)

				cropped_joint.append(joint_img)

				# obtain joint label
				cur_label = torch.zeros(6,)		
				cur_label[joint_encoding[id2joint[joint_id[idx]]]] = 1
				joint_label.append(cur_label)

				# standard categorization of joint relationship
				relation_mask = torch.zeros(self.cluster_cat,)
				cluster_data = self.cluster_assignment[str(img_id)][str(joint_id[idx])]
				cluster_info = tuple([joint_id[idx]] + list(cluster_data.values()))
				cluster_id = self.cluster_category[id2joint[joint_id[idx]]][cluster_info]
				base_residual = 0
				for joint in joint_encoding:
					if joint != id2joint[joint_id[idx]]:
						base_residual += len(self.cluster_category[joint])
					else:
						break
				relation_mask[base_residual+cluster_id] = 1
				relation_label.append(relation_mask)

			# compute the relative location of joints (from cue to target)
			relative_loc = [joint_loc[1][0]-joint_loc[0][0],
							joint_loc[1][1]-joint_loc[0][1]]
			relative_loc = torch.FloatTensor(relative_loc)
			connectivity = torch.Tensor([data['connectivity']])

			return cropped_joint[0], cropped_joint[1], relation_label[0], relation_label[1], joint_label[0], joint_label[1], relative_loc, connectivity

	def __len__(self,):
		return len(self.annotation)


class LSP_pair_generator_ablation(data.Dataset):
	""" Dataloader for ablation study with random pairs (test-only).
	"""
	def __init__(self, img_dir, anno_dir, split='train', 
				joint_size=36, rotate=0):
		""" Initialization function

		Inputs:
			img_dir: Directory storing image files
			anno_dir: Directory storing annotation files
			mode: Specify data split, 'train', 'val', or 'test'
			joint_size: Spatial size of the joint patches
			rotate: Degree of rotation (cue-only) for ablation study
		"""

		self.img_dir = img_dir
		self.split = split
		self.joint_size = joint_size
		self.rotate = rotate
		self.rescaler = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		anno_file = loadmat(os.path.join(anno_dir, 
					'joints_scaled_cropped.mat'))['joints_scaled_cropped']
		self.excluded_data = json.load(open(os.path.join(
									anno_dir, 'excluded_human_data.json')))
		self.cluster_assignment = json.load(open(os.path.join(
									anno_dir, 'cluster_assignment.json')))
		self.cluster_category= np.load(os.path.join(anno_dir, 
								'cluster_assignment_categorized.npy'), allow_pickle=True).item()
		self.image_size = json.load(open(os.path.join(
									anno_dir, 'image_size.json')))
		self.split_info = json.load(open(os.path.join(
									anno_dir, 'split_info_latest.json')))		

		self.init_data(anno_file)

	def init_data(self, anno_file):
		# get the number of clustering categories
		self.cluster_cat = 0
		for joint in self.cluster_category:
			self.cluster_cat += len(self.cluster_category[joint])
		self.cluster2joint = torch.zeros(6, self.cluster_cat)
		residual = 0
		for idx, joint in enumerate(joint_encoding):
			self.cluster2joint[idx, residual:residual+len(self.cluster_category[joint])] =1
			residual += len(self.cluster_category[joint])
			
		# using pre-sampled training/val splits
		if self.split != 'test':
			id_pool = [int(cur)-1 for cur in self.split_info[self.split]]
		else:
			# id_pool = [int(cur)-1 for cur in self.excluded_data]
			id_pool = [int(cur)-1 for cur in self.split_info[self.split]]

		# iterate through all selected images and their corresponding joints
		self.annotation = []
		for img_id in id_pool:
			h, w = self.image_size[str(img_id+1)]
			target_range = range(12)

			for target_id in target_range:
				target_x, target_y = anno_file[0, target_id, img_id], anno_file[1, target_id, img_id]
				if target_x > w or target_y > h:
					continue

				tmp = {'image': img_id+1, 
						'joint': target_id,
						'location': [int(target_y)-1, int(target_x)-1],
						}
				self.annotation.append(tmp)


	def crop_joint(self, image, joint_loc, cue_size=None):
		""" Function for cropping region centered at
			a joint location.

			Inputs:
			image: a tensor containing the whole image.
			joint_loc: location of the joint.
			cue_size: testing with a specific cue size

			Return:
			A tensor containing the cropped joint
		"""

		# determine the boundary of cropped joint
		_, height, width = image.shape
		if cue_size is None:
			y1 = max(0, joint_loc[0]-int(self.joint_size/2))
			y2 = min(joint_loc[0]+int(self.joint_size/2), height)
			x1 = max(0, joint_loc[1]-int(self.joint_size/2))
			x2 = min(joint_loc[1]+int(self.joint_size/2), width)
		else:
			y1 = max(0, joint_loc[0]-int(cue_size/2))
			y2 = min(joint_loc[0]+int(cue_size/2), height)
			x1 = max(0, joint_loc[1]-int(cue_size/2))
			x2 = min(joint_loc[1]+int(cue_size/2), width)

		# crop and rescale
		crop_joint = image[:, y1:y2, x1:x2]
		crop_joint = self.rescaler(crop_joint)

		return crop_joint

	def __getitem__(self, index):

		# selecting target and cue from different images
		data = self.annotation[index]
		img_id = data['image']
		
		cropped_joint = []
		joint_label = []

		# loading target joint data
		img =  Image.open(os.path.join(self.img_dir, 
						'im'+str(img_id).zfill(4))+'_scaled_cropped.jpg'
						).convert('RGB')
		img = transforms.ToTensor()(img)

		# get pre-extracted joints
		target_id = data['joint']
		target_loc = data['location']

		# extract target joint image
		target_joint = self.crop_joint(img, target_loc)
		cropped_joint.append(target_joint)

		# obtain target label
		cur_label = torch.zeros(6,)		
		cur_label[joint_encoding[id2joint[target_id]]] = 1
		joint_label.append(cur_label)		

		# obtaining cue joint data from randomly sampled image
		cue_idx = np.random.choice(np.arange(len(self.annotation)), 1)[0]
		while cue_idx == index:
			cue_idx = np.random.choice(np.arange(len(self.annotation)), 1)[0]

		cue_data = self.annotation[cue_idx]
		cue_img = cue_data['image']

		cue_img =  Image.open(os.path.join(self.img_dir, 
						'im'+str(cue_img).zfill(4))+'_scaled_cropped.jpg'
						).convert('RGB')
		cue_img = transforms.ToTensor()(cue_img)			

		cue_id = cue_data['joint']
		cue_loc = cue_data['location']

		cue_joint = self.crop_joint(cue_img, cue_loc, 60) # ablation with specific cue size
		if self.rotate >0:
			cue_joint = transforms.functional.rotate(cue_joint, self.rotate)
		cropped_joint.append(cue_joint)

		cur_label = torch.zeros(6,)		
		cur_label[joint_encoding[id2joint[cue_id]]] = 1
		joint_label.append(cur_label)		


		# compute the relative location of joints (from cue to target)
		relative_loc = [cue_loc[0]-target_loc[0],
						cue_loc[1]-target_loc[1]]
		relative_loc = torch.FloatTensor(relative_loc)
		connectivity = 1 if cue_id in id_connectivity[target_id] else 0
		connectivity = torch.Tensor([connectivity])

		return cropped_joint[0], cropped_joint[1], joint_label[0], joint_label[1], relative_loc, connectivity

	def __len__(self,):
		return len(self.annotation)


class LSP_pair_generator_all(data.Dataset):
	""" Dataloader for processing LSP joint pairs.
	"""
	def __init__(self, img_dir, anno_dir, split='train', joint_size=36, 
				connected_only=False, ablation=False, rotate=0):
		""" Initialization function

		Inputs:
			img_dir: Directory storing image files
			anno_dir: Directory storing annotation files
			mode: Specify data split, 'train', 'val', or 'test'
			joint_size: Spatial size of the joint patches
			connected_only: Only consider connected pairs or not
			ablation: Testing with an ablation experiment on a specific test set
			rotate: Degree of rotation (cue-only) for ablation study
		"""


		self.img_dir = img_dir
		self.split = split
		self.joint_size = joint_size
		self.connected_only = connected_only
		self.ablation = ablation
		self.rotate = rotate
		self.rescaler = transforms.Compose([
				transforms.Resize((224, 224)),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		anno_file = loadmat(os.path.join(anno_dir, 
					'joints_scaled_cropped.mat'))['joints_scaled_cropped']
		self.excluded_data = json.load(open(os.path.join(
									anno_dir, 'excluded_human_data.json')))
		self.cluster_assignment = json.load(open(os.path.join(
									anno_dir, 'cluster_assignment.json')))
		self.cluster_category= np.load(os.path.join(anno_dir, 
								'cluster_assignment_categorized.npy'), allow_pickle=True).item()
		self.image_size = json.load(open(os.path.join(
									anno_dir, 'image_size.json')))
		self.split_info = json.load(open(os.path.join(
									anno_dir, 'split_info_latest.json')))		
		if self.ablation:
			# self.recorded_acc = json.load(open(os.path.join(anno_dir, 'pred_confidence_36.json')))		
			self.recorded_acc = json.load(open('pred_confidence_baseline_vgg.json'))		

		self.init_data(anno_file)

	def init_data(self, anno_file):
		# get the number of clustering categories
		self.cluster_cat = 0
		for joint in self.cluster_category:
			self.cluster_cat += len(self.cluster_category[joint])
		self.cluster2joint = torch.zeros(6, self.cluster_cat)
		residual = 0
		for idx, joint in enumerate(joint_encoding):
			self.cluster2joint[idx, residual:residual+len(self.cluster_category[joint])] =1
			residual += len(self.cluster_category[joint])

		# using pre-sampled training/val splits
		if self.split != 'test':
			id_pool = [int(cur)-1 for cur in self.split_info[self.split]]
		else:
			# id_pool = [int(cur)-1 for cur in self.excluded_data]
			id_pool = [int(cur)-1 for cur in self.split_info[self.split]]

		# iterate through all selected images and their corresponding joints
		self.annotation = []
		for img_id in id_pool:
			# # for human experiment
			# if self.ablation and str(img_id+1) not in self.recorded_acc:
			# 	continue

			h, w = self.image_size[str(img_id+1)]

			# consider all possible pairs for evaluation
			if not self.ablation:
				target_range = range(12)
				cue_range = range(12)
			else:
				target_range = self.recorded_acc[str(img_id+1)]['low']
				target_range = [int(cur) for cur in target_range] # -1 for human data		
				cue_range = self.recorded_acc[str(img_id+1)]['high']
				cue_range = [int(cur) for cur in cue_range]

			for target_id in target_range:
				target_x, target_y = anno_file[0, target_id, img_id], anno_file[1, target_id, img_id]
				if target_x > w or target_y > h:
					continue

				for cue_id in cue_range:
					# only consider pairs between different joints
					if id2joint[cue_id] == id2joint[target_id]:
						continue

					# remove invalid data
					cue_x, cue_y = anno_file[0, cue_id, img_id], anno_file[1, cue_id, img_id]
					if cue_x > w or cue_y > h:
						continue

					# only consider connected pair
					connected = 1 if cue_id in id_connectivity[target_id] else 0
						
					tmp = {'image': img_id+1, 
							'pair': [target_id, cue_id],
							'location': [[int(target_y)-1, int(target_x)-1],
										[int(cue_y)-1, int(cue_x)-1]],
							'connectivity': connected

							}
					self.annotation.append(tmp)
		# print(len(self.annotation))
		# assert 0

	def crop_joint(self, image, joint_loc, cue_size=None):
		""" Function for cropping region centered at
			a joint location.

			Inputs:
			image: a tensor containing the whole image.
			joint_loc: location of the joint.
			cue_size: testing with a specific cue size

			Return:
			A tensor containing the cropped joint
		"""

		# determine the boundary of cropped joint
		_, height, width = image.shape
		if cue_size is None:
			y1 = max(0, joint_loc[0]-int(self.joint_size/2))
			y2 = min(joint_loc[0]+int(self.joint_size/2), height)
			x1 = max(0, joint_loc[1]-int(self.joint_size/2))
			x2 = min(joint_loc[1]+int(self.joint_size/2), width)
		else:
			y1 = max(0, joint_loc[0]-int(cue_size/2))
			y2 = min(joint_loc[0]+int(cue_size/2), height)
			x1 = max(0, joint_loc[1]-int(cue_size/2))
			x2 = min(joint_loc[1]+int(cue_size/2), width)

		# crop and rescale
		crop_joint = image[:, y1:y2, x1:x2]
		crop_joint = self.rescaler(crop_joint)

		return crop_joint

	def __getitem__(self, index):
		# iterative through all possible pairs for evaluations
		data = self.annotation[index]
		img_id = data['image']

		# loading image
		img =  Image.open(os.path.join(self.img_dir, 
						'im'+str(img_id).zfill(4))+'_scaled_cropped.jpg'
						).convert('RGB')
		img = transforms.ToTensor()(img)

		# get pre-extracted joints
		joint_id = data['pair']
		joint_loc = data['location']

		cropped_joint = []
		joint_label = []
		relation_label = []
		for idx in range(2):
			# obtain joint image
			# joint_img = self.crop_joint(img, joint_loc[idx])

			if idx == 1:
				joint_img = self.crop_joint(img, joint_loc[idx], 36) # ablation with specific cue size
				if self.rotate >0:
					joint_img = transforms.functional.rotate(joint_img, self.rotate)
			else:
				joint_img = self.crop_joint(img, joint_loc[idx])
				# if self.rotate >0:
				# 	joint_img = transforms.functional.rotate(joint_img, self.rotate)

			cropped_joint.append(joint_img)

			# obtain joint label
			cur_label = torch.zeros(6,)		
			cur_label[joint_encoding[id2joint[joint_id[idx]]]] = 1
			joint_label.append(cur_label)

			# standard categorization of joint relationship
			relation_mask = torch.zeros(self.cluster_cat,)
			cluster_data = self.cluster_assignment[str(img_id)][str(joint_id[idx])]
			cluster_info = tuple([joint_id[idx]] + list(cluster_data.values()))
			cluster_id = self.cluster_category[id2joint[joint_id[idx]]][cluster_info]
			base_residual = 0
			for joint in joint_encoding:
				if joint != id2joint[joint_id[idx]]:
					base_residual += len(self.cluster_category[joint])
				else:
					break
			relation_mask[base_residual+cluster_id] = 1
			relation_label.append(relation_mask)

		# compute the relative location of joints (from cue to target)
		relative_loc = [joint_loc[1][0]-joint_loc[0][0],
						joint_loc[1][1]-joint_loc[0][1]]
		relative_loc = torch.FloatTensor(relative_loc)

		if self.split != 'train':
			connectivity = torch.Tensor([data['connectivity']])
			return cropped_joint[0], cropped_joint[1], relation_label[0], relation_label[1], joint_label[0], joint_label[1], relative_loc, connectivity
		else:
			return cropped_joint[0], cropped_joint[1], relation_label[0], relation_label[1], joint_label[0], joint_label[1], relative_loc

	def __len__(self,):
		return len(self.annotation)


class LSP_pair_revision(data.Dataset):
	""" Dataloader for processing LSP joint pairs for paper revision.
	"""
	def __init__(self, anno_dir, split='train', layout='ori', test_id=None, 
				connectivity=None, target_size=None, cue_size=None, rotation=None):
		""" Initialization function

		Inputs:
			anno_dir: Directory storing annotation files
			split: Specify data split, 'train', 'val', or 'test'
			layout: Original or side-by-side layout
			test_id: Specifying the experiment id (1 or 2)
			connectivity: Evaluation with connected or unconnected joints
			target_size: Target size for evaluation
			cue_size: Cue size for evaluation
			rotation: rotated degree
		"""


		self.split = split
		self.layout = layout
		self.test_id = test_id
		self.target_size = target_size
		self.cue_size = cue_size
		self.rotation = rotation
		self.connectivity = connectivity
		self.transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
			])
		
		self.init_data(anno_dir)

	def init_data(self, anno_dir):
		self.img_pool = []
		self.annotation = []

		if self.split != 'test':
			self.anno_dir = os.path.join(anno_dir, 'train_pairs_all')
			tmp_pool = json.load(open(os.path.join(self.anno_dir, 
										'split_info_revision_all.json')))[self.split]
			invalid = json.load(open(os.path.join(anno_dir, 'invalid_train_vgg.json')))
			for cur_img in tmp_pool:
				if cur_img in invalid:
					continue
				target_id, cue_id = cur_img.split('_')[1], cur_img.split('_')[3]
				self.img_pool.append(cur_img)
				self.annotation.append([int(target_id)-1, int(cue_id)-1])
		else:
			if self.test_id == 'exp1':
				self.anno_dir = os.path.join(anno_dir, 'test_pairs', 
									'exp1_size', self.layout+'_layout')
				tmp_pool = [os.path.basename(
							cur) for cur in glob(
							os.path.join(self.anno_dir, 
								'img_pairs', self.connectivity, '*.jpg'))]
				for cur_img in tmp_pool:
					# ignore opposite side-by-side
					if self.layout == 'sbs' and 'oppo' in cur_img:
						continue
					if self.layout == 'sbs':
						target_size, cue_size = cur_img.split('_')[2], cur_img.split('_')[4]
					else:
						target_size, cue_size = cur_img.split('_')[2], cur_img.split('_')[4][:-4]
					
					# print(target_size, self.target_size)
					# print(cue_size, self.cue_size)
					# print('-----------')
					# assert 0

					if target_size != self.target_size or cue_size != self.cue_size:
						continue
					target_id, cue_id = cur_img.split('_')[1], cur_img.split('_')[3]
					self.img_pool.append(cur_img)
					self.annotation.append([int(target_id)-1, int(cue_id)-1])
			else:
				self.anno_dir = os.path.join(anno_dir, 'test_pairs', 
									'exp2_rotation')
				tmp_pool = [os.path.basename(
							cur) for cur in glob(
							os.path.join(self.anno_dir, 
								'img_pairs', self.connectivity, '*.jpg'))]
				for cur_img in tmp_pool:
					rotate_degree = cur_img.split('_')[-1][:-5]
					# print(rotate_degree, self.rotation)
					if rotate_degree != self.rotation:
						continue
					target_id, cue_id = cur_img.split('_')[1], cur_img.split('_')[3]
					self.img_pool.append(cur_img)
					self.annotation.append([int(target_id)-1, int(cue_id)-1])

	def __getitem__(self, index):
		# iterative through all possible pairs for evaluations
		img_id = self.img_pool[index]

		# loading image and target/cue mask
		if self.split != 'test':
			img =  Image.open(os.path.join(self.anno_dir, 
							'img_pairs', img_id)).convert('RGB')
			img = self.transform(img)
			target_mask = cv2.imread(os.path.join(self.anno_dir, 'tar_masks', 
									img_id[:-4]+'_tar_mask.jpg'))
			cue_mask = cv2.imread(os.path.join(self.anno_dir, 'cue_masks', 
									img_id[:-4]+'_cue_mask.jpg'))
		else:
			img =  Image.open(os.path.join(self.anno_dir, 
							'img_pairs', self.connectivity, img_id)).convert('RGB')
			img = self.transform(img)
			target_mask = cv2.imread(os.path.join(self.anno_dir, 'tar_masks', 
							self.connectivity, img_id[:-4]+'_tar_mask.jpg'))
			cue_mask = cv2.imread(os.path.join(self.anno_dir, 'cue_masks', 
							self.connectivity, img_id[:-4]+'_cue_mask.jpg'))

		# process the target/cue mask for classification
		target_mask = target_mask.astype('float32')[:,:,0]/255
		cue_mask = cue_mask.astype('float32')[:,:,0]/255
		union_mask = np.minimum(target_mask+cue_mask, 1)
		union_mask[union_mask>0.5] = 1
		union_mask[union_mask!=1] = 0

		target_mask = cv2.resize(target_mask, (10, 10)) # 10x10 for resnet, 9x9 for vgg
		target_mask[target_mask>0] = 1
		# target_mask[target_mask!=1] = 0
		cue_mask = cv2.resize(cue_mask, (10, 10))
		cue_mask[cue_mask>0] = 1
		# cue_mask[cue_mask!=1] = 0

		# load label
		target_label = torch.zeros(6,)		
		target_label[joint_encoding[id2joint[self.annotation[index][0]]]] = 1
		cue_label = torch.zeros(6,)		
		cue_label[joint_encoding[id2joint[self.annotation[index][1]]]] = 1

		return img, union_mask, target_mask, cue_mask, target_label, cue_label

	def __len__(self,):
		return len(self.img_pool)