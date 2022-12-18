import torch
import torch.nn as nn
import torch.nn.functional as F

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
		
		# predicting the pairwise relationship
		rel_prediction = torch.sigmoid(self.rel_classifier(cls_feat))
		# using joint prediction as a masking
		batch = len(joint_prediction)
		rel_prediction = rel_prediction.view(batch, 6, -1)
		rel_prediction = rel_prediction* joint_prediction.unsqueeze(-1)

		return joint_prediction, rel_prediction.view(batch, -1)
