import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_ori import resnet50

class PoseNet(nn.Module):
	""" Re-implementation of the pose estimation model
	introduced in paper (hyperparameters are fixed):
	https://proceedings.neurips.cc/paper/2014/file/8b6dd7db9af49e67306feb59a8bdc52c-Paper.pdf
	"""

	def __init__(self, num_cluster=11):
		super(PoseNet, self).__init__()	
		self.encoder = nn.Sequential(
						nn.Conv2d(3, 32, kernel_size=5, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.MaxPool2d(kernel_size=2, stride=2),
						nn.Conv2d(32, 128, kernel_size=3, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.MaxPool2d(kernel_size=2, stride=2),
						nn.Conv2d(128, 128, kernel_size=3, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.Conv2d(128, 128, kernel_size=3, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.Conv2d(128, 128, kernel_size=3, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.MaxPool2d(kernel_size=3, stride=3),
						)

		self.decoder = nn.Sequential(
						nn.Dropout(0.5),
						nn.Linear(1152, 512), # originally 10368, 4096
						nn.ReLU(),
						nn.Dropout(0.5),
						nn.Linear(512, 512),
						nn.ReLU(),
						)

		self.joint_classifier = nn.Linear(512, 6)
		self.rel_classifier = nn.Linear(512, 6*5*num_cluster)

	def forward(self, img):
		""" A straightforward inference procedure.
		"""
		img_feat = self.encoder(img)
		cls_feat = self.decoder(img_feat.view(img_feat.shape[0], -1))

		# predicting the joint category
		joint_prediction = F.softmax(self.joint_classifier(cls_feat), dim=-1)
		# return joint_prediction

		# predicting the pairwise relationship
		rel_prediction = torch.sigmoid(self.rel_classifier(cls_feat))
		# using joint prediction as a masking
		batch = len(joint_prediction)
		rel_prediction = rel_prediction.view(batch, 6, -1)
		rel_prediction = rel_prediction* joint_prediction.unsqueeze(-1)

		return joint_prediction, rel_prediction.view(batch, -1)


class PoseNet_slim(nn.Module):
	""" Re-implementation of the pose estimation model
	introduced in paper (hyperparameters are fixed):
	https://proceedings.neurips.cc/paper/2014/file/8b6dd7db9af49e67306feb59a8bdc52c-Paper.pdf

	Different from the standard PoseNet, here the joint categorization 
	is a direct conversion of relationship prediction.
	"""

	def __init__(self, num_cluster=11):
		super(PoseNet_slim, self).__init__()	
		self.encoder = nn.Sequential(
						nn.Conv2d(3, 32, kernel_size=5, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.MaxPool2d(kernel_size=2, stride=2),
						nn.Conv2d(32, 128, kernel_size=3, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.MaxPool2d(kernel_size=2, stride=2),
						nn.Conv2d(128, 128, kernel_size=3, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.Conv2d(128, 128, kernel_size=3, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.Conv2d(128, 128, kernel_size=3, 
							padding='same', bias=True),
						nn.ReLU(),
						nn.MaxPool2d(kernel_size=3, stride=3),
						)

		self.decoder = nn.Sequential(
						nn.Dropout(0.5),
						nn.Linear(1152, 512), # originally 10368, 4096
						nn.ReLU(),
						nn.Dropout(0.5),
						nn.Linear(512, 512),
						nn.ReLU(),
						)

		self.rel_classifier = nn.Linear(512, 6*5*num_cluster)

	def forward(self, img):
		""" A straightforward inference procedure.
		"""
		img_feat = self.encoder(img)
		cls_feat = self.decoder(img_feat.view(img_feat.shape[0], -1))

		# predicting the pairwise relationship
		batch = len(img_feat)
		rel_prediction = torch.sigmoid(self.rel_classifier(cls_feat))
		joint_prediction = F.softmax(
							rel_prediction.view(batch, 6, -1).sum(-1), dim=-1)

		return joint_prediction, rel_prediction


class ResNet50(nn.Module):
	def __init__(self, use_rel=False, num_cluster=None, cluster2joint=None):
		super(ResNet50, self).__init__()	
		self.backbone = resnet50(pretrained=True)
		# self.dilate_resnet(self.backbone)
		self.backbone = nn.Sequential(*list(
								self.backbone.children())[:-2])
		self.use_rel = use_rel
		if self.use_rel:
			self.classifer = nn.Linear(2048, num_cluster)
			self.cluster2joint = nn.Parameter(cluster2joint, requires_grad=False)
		else:
			self.classifer = nn.Linear(2048, 6)

	def dilate_resnet(self, resnet): 
		""" Converting standard ResNet50 into a dilated one.
		"""
		resnet.layer3[0].conv1.stride = 1
		resnet.layer3[0].downsample[0].stride = 1
		resnet.layer4[0].conv1.stride = 1
		resnet.layer4[0].downsample[0].stride = 1

	def forward(self, img, feat_only=False):
		img_feat = self.backbone(img)
		batch, c, h, w = img_feat.shape
		img_feat = img_feat.view(batch, c, h*w).mean(-1)

		if feat_only:
			return img_feat
		else:
			if not self.use_rel:
				joint_pred = F.softmax(self.classifer(F.dropout(img_feat, 0.5)), dim=-1)
				return joint_pred
			else:
				rel_pred = F.softmax(self.classifer(F.dropout(img_feat, 0.5)), dim=-1)
				joint_pred = rel_pred.unsqueeze(1)*self.cluster2joint.unsqueeze(0)
				joint_pred = joint_pred.sum(-1)
				return joint_pred, rel_pred

class Pairwise_Estimation(nn.Module):
	""" Re-implementation of the pose estimation model (second phase)
	introduced in paper (hyperparameters are fixed):
	https://proceedings.neurips.cc/paper/2014/file/8b6dd7db9af49e67306feb59a8bdc52c-Paper.pdf
	"""

	def __init__(self, backbone, hidden_size=100, use_rel=False, num_rel=None, 
					freeze_backbone=False, use_pos=False):
		""" Model initialization.

			Input:
				backbone: pretrained backbone model
				hidden_size: size of hidden layer
				use_rel: using relation prediction for backbone or not
				num_rel: number of candidate relation
				freeze_backbone: freezing backbone weights or not
				use_pos: using relative position information or not
		"""
		super(Pairwise_Estimation, self).__init__()
		
		self.backbone = backbone
		self.freeze_backbone = freeze_backbone
		if freeze_backbone:
			for para in self.backbone.parameters():
				para.requires_grad = False

		self.use_rel = use_rel
		self.use_pos = use_pos
		if use_rel:
			self.target_encoder = nn.Linear(num_rel, hidden_size)
			self.cue_encoder = nn.Linear(num_rel, hidden_size)
			# self.target_encoder = nn.Linear(2048, hidden_size)
			# self.cue_encoder = nn.Linear(2048, hidden_size)

		else:		
			self.target_encoder = nn.Linear(6, hidden_size)
			self.cue_encoder = nn.Linear(6, hidden_size)

		if self.use_pos:
			self.pos_encoder = nn.Linear(2, hidden_size)
			self.final_cls = nn.Linear(3*hidden_size, 6)
		else:
			self.final_cls = nn.Linear(2*hidden_size, 6)

		self.dp = nn.Dropout(0.5)

	def forward(self, target_img, cue_img, relative_pos):
		if self.use_rel:
			target_joint, target_pred = self.backbone(target_img)
			cue_joint, cue_pred = self.backbone(cue_img)
			# target_pred = self.backbone(target_img, feat_only=True)
			# cue_pred = self.backbone(cue_img, feat_only=True)

		else:
			target_pred = self.backbone(target_img)
			cue_pred = self.backbone(cue_img)

		target_feat = torch.relu(self.target_encoder(target_pred))
		cue_feat = torch.relu(self.target_encoder(cue_pred))

		if self.use_pos:
			pos_feat = torch.relu(self.pos_encoder(relative_pos))
			fuse_feat = torch.cat([target_feat, cue_feat, pos_feat], dim=-1)
		else:
			fuse_feat = torch.cat([target_feat, cue_feat], dim=-1)

		final_pred = F.softmax(self.final_cls(self.dp(fuse_feat)), dim=-1)
		if self.freeze_backbone:
			return final_pred
		else:
			return final_pred, target_pred, cue_pred, target_joint, cue_joint
