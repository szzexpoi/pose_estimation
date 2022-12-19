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

class LSP_joint_generator(data.Dataset):
	""" Dataloader for processing LSP joint data
	"""
	def __init__(self, img_dir, anno_dir, split='train', joint_size=36, num_cluster=11):
		""" Initialization function

		Inputs:
			img_dir: Directory storing image files
			anno_dir: Directory storing annotation files
			mode: Specify data split, 'train', 'val', or 'test'
			joint_size: Spatial size of the joint patches
			num_cluster: Number of clusters of pairwise relationship
		"""

		self.img_dir = img_dir
		self.split = split
		self.joint_size = joint_size
		self.num_cluster = num_cluster
		self.rescaler = transforms.Resize((36, 36))

		anno_file = loadmat(os.path.join(anno_dir,
					'joints_scaled_cropped.mat'))['joints_scaled_cropped']
		self.excluded_data = json.load(open(os.path.join(
									anno_dir, 'excluded_human_data.json')))
		self.cluster_assignment = json.load(open(os.path.join(
									anno_dir, 'cluster_assignment.json')))
		self.image_size = json.load(open(os.path.join(
									anno_dir, 'image_size.json')))
		self.init_data(anno_file)

	def init_data(self, anno_file):
		# the data is splited by half for training and evaluation
		id_pool = range(1000) if self.split == 'train' else range(1000, 2000)

		# iterate through all selected images and their corresponding joints
		self.annotation = []
		for img_id in id_pool:
			if img_id+1 in self.excluded_data:
				continue

			h, w = self.image_size[str(img_id+1)]

			# only considering 6 pairs of joints
			for joint_id in range(12):
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
		# print(y1, y2, x1, x2, crop_joint.shape)
		crop_joint = self.rescaler(crop_joint)
		# cv2.imwrite('check.jpg', crop_joint.permute(1,2,0).data.numpy()*255)
		# assert 0


		return crop_joint

	def __getitem__(self, index):
		data = self.annotation[index]
		img_id = data['image']
		joint_id = data['joint']
		joint_loc = data['location']

		# loading image
		img =  Image.open(os.path.join(self.img_dir,
						'im'+str(img_id).zfill(4))+'_scaled_cropped.jpg'
						).convert('RGB')
		img = transforms.ToTensor()(img)

		# obtaining the cropped region centered at joint
		joint_img = self.crop_joint(img, joint_loc)

		# category of joint
		joint_label = torch.zeros(6,)
		joint_label[joint_encoding[id2joint[joint_id]]] = 1

		# a multi-label mask encoding relationship (6 joints x 5 other joints x number of clusters)
		relation_mask = torch.zeros(6*5*self.num_cluster,)
		cluster_data = self.cluster_assignment[str(img_id)][str(joint_id)]

		# residual for joint category
		base_residual = joint_encoding[id2joint[joint_id]]*5*self.num_cluster
		# residual for one joint to the others
		pair_residual = 0

		for target_joint in joint_encoding:
			if target_joint == id2joint[joint_id]:
				continue
			for target_id in joint2id[target_joint]:
				cluster_label = cluster_data[str(target_id)]
				cluster_label += base_residual+pair_residual
				relation_mask[cluster_label] = 1
			pair_residual += self.num_cluster

		return joint_img, joint_label, relation_mask

	def __len__(self,):
		return len(self.annotation)
