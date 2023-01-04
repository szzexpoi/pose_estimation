import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import LSP_pair_generator
from model import ResNet50, Pairwise_Estimation
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
parser.add_argument('--anno_dir', type=str, default='./data', help='Directory to annotation files')
parser.add_argument('--img_dir', type=str, default='./data/img_scaled_cropped', help='Directory to image files')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory for saving checkpoint')
parser.add_argument('--weights', type=str, default=None, help='Trained model to be loaded (default: None)')
parser.add_argument('--pretrained', type=str, default=None, help='Pretrained model to be loaded (default: None)')
parser.add_argument('--epoch', type=int, default=60, help='Defining maximal number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Defining initial learning rate (default: 1e-3)')
parser.add_argument('--batch_size', type=int, default=128, help='Defining batch size for training (default: 32)')
parser.add_argument('--clip', type=float, default=0, help='Gradient clipping to prevent gradient explode (default: 0)')
parser.add_argument('--joint_size', type=int, default=32, help='Defining size of the patches for joints')
parser.add_argument('--alpha', type=float, default=1, help='balancing factor for training')
parser.add_argument('--use_rel', type=bool, default=False, help='using clustering relationship or not for ResNet')
parser.add_argument('--use_pos', type=bool, default=False, help='using relative position or not for ResNet')
parser.add_argument('--hidden_size', type=int, default=100, help='Defining size of hidden layer')
parser.add_argument('--connected_only', type=bool, default=False, help='Only considering connected joints')
parser.add_argument('--freeze_backbone', type=bool, default=False, help='Using fixed backbone')
parser.add_argument('--ablation', type=bool, default=False, help='Using special test cases for ablation or not')
parser.add_argument('--rotate', type=int, default=0, help='Degree of rotation for ablation study')
parser.add_argument('--cue_size', type=int, default=None, help='Using specific cue size for ablation study')

args = parser.parse_args()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(init_lr,optimizer, epoch):
    "adatively adjust lr based on epoch"
    lr = init_lr * (0.25 ** int((epoch+1)/30)) #previously 0.25/20

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)

    # define dataloader
    train_data = LSP_pair_generator(args.img_dir, args.anno_dir, 'train', args.joint_size, args.connected_only)
    val_data = LSP_pair_generator(args.img_dir, args.anno_dir, 'val', args.joint_size, args.connected_only)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=12)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)

    # initialize model
    if not args.use_rel:
        backbone = ResNet50()
    else:
        backbone = ResNet50(args.use_rel, train_data.cluster_cat,
                    train_data.cluster2joint)
    backbone.load_state_dict(torch.load(args.pretrained), strict=True)
    model = Pairwise_Estimation(backbone, args.hidden_size, args.use_rel,
                            train_data.cluster_cat, args.freeze_backbone, args.use_pos)
    model = model.cuda()

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                        lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) #1e-8

    def train(iteration):
        """ Training process for a single epoch.
        """
        model.train()
        avg_joint_loss = 0
        avg_rel_loss = 0

        for batch_idx,(target_img, cue_img, target_rel, cue_rel, target_label, cue_label, relative_pos) in enumerate(trainloader):
            target_img, cue_img, target_rel, cue_rel, target_label, cue_label, relative_pos = target_img.cuda(), cue_img.cuda(), target_rel.cuda(), cue_rel.cuda(), target_label.cuda(), cue_label.cuda(), relative_pos.cuda()
            optimizer.zero_grad()

            if args.freeze_backbone:
                target_pred = model(target_img, cue_img, relative_pos)
                joint_loss = cross_entropy(target_pred, target_label)
                loss = joint_loss
            else:
                target_pred, target_rel_pred, cue_rel_pred, target_joint, cue_joint = model(target_img, cue_img, relative_pos)
                joint_loss = cross_entropy(target_pred, target_label) + cross_entropy(cue_joint, cue_label)
                rel_loss = cross_entropy(target_rel_pred, target_rel) + cross_entropy(cue_rel_pred, cue_rel)
                loss = joint_loss + args.alpha*rel_loss

            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()

            avg_joint_loss = (avg_joint_loss*np.maximum(0, batch_idx) +
                        joint_loss.data.cpu().numpy())/(batch_idx+1)

            if not args.freeze_backbone:
                avg_rel_loss = (avg_rel_loss*np.maximum(0, batch_idx) +
                            rel_loss.data.cpu().numpy())/(batch_idx+1)

            if batch_idx%25 == 0:
                with tf_summary_writer.as_default():
                    tf.summary.scalar('average joint loss', avg_joint_loss, step=iteration)
                    tf.summary.scalar('average relationship loss', avg_rel_loss, step=iteration)

            iteration += 1

        return iteration


    def test(iteration):
        """ Function for validation.
        """
        model.eval()

        # we evaluate the performance for both joint categorization and relationship prediction
        joint_acc = []
        rel_acc = []

        with torch.no_grad():
            for batch_idx,(target_img, cue_img, target_rel, cue_rel, target_label, cue_label, relative_pos, _) in enumerate(valloader):
                target_img, cue_img, relative_pos = target_img.cuda(), cue_img.cuda(), relative_pos.cuda()

                # obtain prediction
                if args.freeze_backbone:
                    target_pred = model(target_img, cue_img, relative_pos)
                else:
                    target_pred, _, _, _, _ = model(target_img, cue_img, relative_pos)


                # convert prediction and label
                target_pred = target_pred.argmax(-1).data.cpu().numpy()
                target_label = target_label.argmax(-1).data.numpy()

                # compute accuracy for joint categorization
                joint_acc.extend(target_pred==target_label)


        joint_acc = np.mean(np.array(joint_acc))

        with tf_summary_writer.as_default():
            tf.summary.scalar('Joint-Accuracy', joint_acc, step=iteration)

        return joint_acc


    #main loop for training:
    print('Start training model')
    iteration = 0
    val_acc = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr,optimizer, epoch)
        iteration = train(iteration)
        if (epoch+1)%3 == 0:
            cur_score = test(iteration)

            #save the best check point and latest checkpoint
            if cur_score > val_acc:
                torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
                val_acc = cur_score
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))


def eval():
    # define dataloader
    test_data = LSP_pair_generator(args.img_dir, args.anno_dir, 'test', args.joint_size,
                                args.connected_only, args.ablation, args.rotate, args.cue_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)

    # initialize model
    if not args.use_rel:
        backbone = ResNet50()
    else:
        backbone = ResNet50(args.use_rel, test_data.cluster_cat,
                    test_data.cluster2joint)
    backbone.load_state_dict(torch.load(args.pretrained))
    model = Pairwise_Estimation(backbone, args.hidden_size, args.use_rel, test_data.cluster_cat,
                                args.freeze_backbone, args.use_pos)

    model.load_state_dict(torch.load(args.weights))
    model = model.cuda()
    model.eval()

    # Evaluate the performance for both joint categorization and relationship prediction
    joint_acc = []
    connected_acc = 0
    connected_count = 0
    unconnected_acc = 0
    unconnected_count = 0

    with torch.no_grad():
        for batch_idx,(target_img, cue_img, target_rel, cue_rel, target_label, cue_label, relative_pos, connectivity) in enumerate(testloader):
            target_img, cue_img, relative_pos = target_img.cuda(), cue_img.cuda(), relative_pos.cuda()
            target_pred = model(target_img, cue_img, relative_pos)

            # convert prediction and label
            target_pred = target_pred.argmax(-1).data.cpu().numpy()
            target_label = target_label.argmax(-1).data.numpy()
            connectivity = connectivity.squeeze(-1).data.numpy()

            # compute accuracy for joint categorization
            joint_acc.extend(target_pred==target_label)
            connected_acc += ((target_pred==target_label)*connectivity).sum()
            connected_count += connectivity.sum()
            unconnected_acc += ((target_pred==target_label)*(1-connectivity)).sum()
            unconnected_count += (1-connectivity).sum()

    joint_acc = np.mean(np.array(joint_acc))
    print('Test joint accuracy is %.3f' %joint_acc)
    print('Connected joint accuracy is %.3f' %(connected_acc/connected_count))
    print('Unconnected joint accuracy is %.3f' %(unconnected_acc/unconnected_count))

if args.mode == 'train':
    main()
else:
    eval()
