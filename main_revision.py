import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import LSP_pair_revision
from model import ResNet50_rev, VGG19_rev
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
import numpy as np
import cv2
import argparse
import os
import time
import gc
import tensorflow as tf
from loss import bce_loss, cross_entropy
import json

parser = argparse.ArgumentParser(description='Pairwise pose estimation on the LSP dataset')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--anno_dir', type=str, default='./data/revision_data', help='Directory to annotation files')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory for saving checkpoint')
parser.add_argument('--weights', type=str, default=None, help='Trained model to be loaded (default: None)')
parser.add_argument('--epoch', type=int, default=60, help='Defining maximal number of epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Defining initial learning rate (default: 1e-3)')
parser.add_argument('--batch_size', type=int, default=32, help='Defining batch size for training (default: 32)')
parser.add_argument('--clip', type=float, default=0, help='Gradient clipping to prevent gradient explode (default: 0)')
parser.add_argument('--target_size', type=str, default=60, help='Defining size of the patches for targets (evaluation)')
parser.add_argument('--cue_size', type=str, default=60, help='Defining size of the patches for cues (evaluation)')
parser.add_argument('--connectivity', type=str, default='connected', help='Connectivity for evaluation')
parser.add_argument('--rotate', type=str, default=None, help='Degree of rotation for evaluation')
parser.add_argument('--layout', type=str, default='ori', help='Joint layout for evaluation, ori or sbs')
parser.add_argument('--exp_id', type=str, default='exp1', help='Experiment ID for evaluation, exp1 or exp2')

args = parser.parse_args()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(init_lr,optimizer, epoch):
    "adatively adjust lr based on epoch"
    lr = init_lr * (0.1 ** int((epoch+1)/3)) #previously 0.25/20

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)

    # define dataloader
    train_data = LSP_pair_revision(args.anno_dir, 'train')
    val_data = LSP_pair_revision(args.anno_dir, 'val')
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=12)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)

    # model = ResNet50_rev()
    model = VGG19_rev()

    model = model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) #1e-8

    def train(iteration):
        """ Training process for a single epoch.
        """
        model.train()
        avg_loss = 0

        for batch_idx,(img, union_mask, target_mask,
                    cue_mask, target_label, cue_label) in enumerate(trainloader):
            img, union_mask, target_mask, cue_mask, target_label, cue_label = img.cuda(), union_mask.cuda(), target_mask.cuda(), cue_mask.cuda(), target_label.cuda(), cue_label.cuda()
            optimizer.zero_grad()

            target_pred, cue_pred = model(img, union_mask, target_mask, cue_mask)
            target_loss = cross_entropy(target_pred, target_label)
            cue_loss = cross_entropy(cue_pred, cue_label)
            loss = target_loss + cue_loss
            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()

            avg_loss = (avg_loss*np.maximum(0, batch_idx) +
                        loss.data.cpu().numpy())/(batch_idx+1)

            if batch_idx%25 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('average loss', avg_loss, step=iteration)
            iteration += 1

        return iteration


    def test(iteration):
        """ Function for validation.
        """
        model.eval()

        # we evaluate the performance for both joint categorization and relationship prediction
        joint_acc = []
        joint_acc_cue = []

        with torch.no_grad():
            for batch_idx,(img, union_mask, target_mask,
                    cue_mask, target_label, cue_label) in enumerate(valloader):
                img, union_mask, target_mask, cue_mask  = img.cuda(), union_mask.cuda(), target_mask.cuda(), cue_mask.cuda()

                target_pred, cue_pred = model(img, union_mask, target_mask, cue_mask)

                # convert prediction and label
                target_pred = target_pred.argmax(-1).data.cpu().numpy()
                target_label = target_label.argmax(-1).data.numpy()
                cue_pred = cue_pred.argmax(-1).data.cpu().numpy()
                cue_label = cue_label.argmax(-1).data.numpy()

                # compute accuracy for joint categorization
                joint_acc.extend(target_pred==target_label)
                joint_acc_cue.extend(cue_pred==cue_label)


        joint_acc = np.mean(np.array(joint_acc))
        joint_acc_cue = np.mean(np.array(joint_acc_cue))

        with tf_summary_writer.as_default():
            tf.summary.scalar('Joint-Accuracy', joint_acc, step=iteration)
            tf.summary.scalar('Joint-Accuracy-cue', joint_acc_cue, step=iteration)

        return joint_acc


    #main loop for training:
    print('Start training model')
    iteration = 0
    val_acc = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr,optimizer, epoch)
        iteration = train(iteration)
        if (epoch+1)%1 == 0:
            cur_score = test(iteration)

            #save the best check point and latest checkpoint
            if cur_score > val_acc:
                torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
                val_acc = cur_score
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))


def eval():
    # define dataloader
    test_data = LSP_pair_revision(args.anno_dir, 'test', args.layout, args.exp_id,
                                args.connectivity, args.target_size, args.cue_size,
                                args.rotate)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)

    # initialize model
    model = ResNet50_rev()
    # model = VGG19_rev()

    model.load_state_dict(torch.load(args.weights))
    model = model.cuda()
    model.eval()

    # Evaluate the performance for both joint categorization and relationship prediction
    joint_acc = []
    joint_acc_cue = []

    with torch.no_grad():
        for batch_idx,(img, union_mask, target_mask,
                cue_mask, target_label, cue_label) in enumerate(testloader):
            img, union_mask, target_mask, cue_mask  = img.cuda(), union_mask.cuda(), target_mask.cuda(), cue_mask.cuda()

            target_pred, cue_pred = model(img, union_mask, target_mask, cue_mask)
            target_pred = target_pred.argmax(-1).data.cpu().numpy()
            target_label = target_label.argmax(-1).data.numpy()
            cue_pred = cue_pred.argmax(-1).data.cpu().numpy()
            cue_label = cue_label.argmax(-1).data.numpy()

            # compute accuracy for joint categorization
            joint_acc.extend(target_pred==target_label)
            joint_acc_cue.extend(cue_pred==cue_label)

    joint_acc = np.mean(np.array(joint_acc))
    joint_acc_cue = np.mean(np.array(joint_acc_cue))
    print('Test joint accuracy is %.3f' %joint_acc)

if args.mode == 'train':
    main()
else:
    eval()
