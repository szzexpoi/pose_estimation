import torch
EPSILON = 1e-15

def bce_loss(input,target):
	loss = -target*torch.log(torch.clamp(input, min=EPSILON, max=1)
		) - (1-target)*torch.log(torch.clamp(1-input, min=EPSILON, max=1))

	return loss.mean()

def cross_entropy(input,target):
	loss = -(target*torch.log(torch.clamp(input,min=EPSILON,max=1))).sum(-1)
	return loss.mean()