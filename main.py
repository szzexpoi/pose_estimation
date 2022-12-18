import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import LSP_joint_generator
from model import PoseNet
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

parser = argparse.ArgumentParser(description='Articulated pose estimation on the LSP dataset')
parser.add_argument('--mode', type=str, default='train', help='Selecting running mode (default: train)')
parser.add_argument('--anno_dir', type=str, default='./data', help='Directory to annotation files')
parser.add_argument('--img_dir', type=str, default='./data/img_scaled_cropped', help='Directory to image files')
parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory for saving checkpoint')
parser.add_argument('--weights', type=str, default=None, help='Trained model to be loaded (default: None)')
parser.add_argument('--epoch', type=int, default=120, help='Defining maximal number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Defining initial learning rate (default: 1e-3)')
parser.add_argument('--batch_size', type=int, default=128, help='Defining batch size for training (default: 32)')
parser.add_argument('--clip', type=float, default=0, help='Gradient clipping to prevent gradient explode (default: 0)')
parser.add_argument('--joint_size', type=int, default=32, help='Defining size of the patches for joints')
parser.add_argument('--num_cluster', type=int, default=11, help='Defining number of clusters for pairwise relationship')
parser.add_argument('--alpha', type=float, default=1, help='balancing factor for training')

args = parser.parse_args()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(init_lr,optimizer, epoch):
    "adatively adjust lr based on epoch"
    lr = init_lr * (0.25 ** int((epoch+1)/50)) #previously 0.25/40

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)

    # define dataloader
    train_data = LSP_joint_generator(args.img_dir, args.anno_dir, 'train', args.joint_size, args.num_cluster)
    val_data = LSP_joint_generator(args.img_dir, args.anno_dir, 'val', args.joint_size, args.num_cluster)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                    shuffle=True, num_workers=12)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)

    # initialize model
    model = PoseNet(args.num_cluster)
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

        for batch_idx,(img, joint_label, rel_label) in enumerate(trainloader):
            img, joint_label, rel_label = img.cuda(), joint_label.cuda(), rel_label.cuda()
            optimizer.zero_grad()

            img = img.data.cpu().numpy()
            pred_joint, pred_rel = model(img)

            joint_loss = cross_entropy(pred_joint, joint_label)
            rel_loss = bce_loss(pred_rel, rel_label)
            loss = joint_loss + args.alpha*rel_loss
            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()

            avg_joint_loss = (avg_joint_loss*np.maximum(0, batch_idx) +
                        joint_loss.data.cpu().numpy())/(batch_idx+1)
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
        rel_tp = 0
        rel_fp = 0
        rel_pos = 0

        with torch.no_grad():
            for batch_idx,(img, joint_label, rel_label) in enumerate(valloader):
                img = img.cuda()

                # obtain prediction
                pred_joint, pred_rel = model(img)
                # pred_joint = model(img)

                # convert prediction and label
                pred_joint = pred_joint.argmax(-1).data.cpu().numpy()
                pred_rel = pred_rel.data.cpu().numpy()
                pred_rel[pred_rel>=0.5] = 1
                pred_rel[pred_rel<0.5] = 0
                joint_label = joint_label.argmax(-1).data.numpy()
                rel_label = rel_label.data.numpy()

                # compute accuracy for joint categorization
                joint_acc.extend(pred_joint==joint_label)

                # compute True/False positive for relationship prediction
                rel_pos += np.sum(rel_label)
                rel_tp += np.sum(rel_label*pred_rel)
                rel_fp += np.sum((1-rel_label)*pred_rel)

        joint_acc = np.mean(np.array(joint_acc))
        precision = rel_tp/(rel_tp+rel_fp+1e-5)
        recall = rel_tp/(rel_pos+1e-5)
        f1_score = 2*(precision*recall)/(precision+recall+1e-5)

        with tf_summary_writer.as_default():
            tf.summary.scalar('Joint-Accuracy', joint_acc, step=iteration)
            tf.summary.scalar('Relation-Precision', precision, step=iteration)
            tf.summary.scalar('Relation-Recall', recall, step=iteration)
            tf.summary.scalar('Relation-F1 score', f1_score, step=iteration)

        return joint_acc


    #main loop for training:
    print('Start training model')
    iteration = 0
    val_acc = 0
    for epoch in range(args.epoch):
        adjust_learning_rate(args.lr,optimizer, epoch)
        iteration = train(iteration)
        cur_score = test(iteration)

        #save the best check point and latest checkpoint
        if cur_score > val_acc:
            torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
            val_acc = cur_score
        torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))


def eval():
    # define dataloader
    test_data = LSP_joint_generator(args.img_dir, args.anno_dir, 'test', args.joint_size, args.num_cluster)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                    shuffle=False, num_workers=4)

    # initialize model
    model = PoseNet(args.num_cluster)
    model = model.cuda()
    model.eval()

    # Evaluate the performance for both joint categorization and relationship prediction
    joint_acc = []
    rel_tp = 0
    rel_fp = 0
    rel_pos = 0

    with torch.no_grad():
        for batch_idx,(img, joint_label, rel_label) in enumerate(testloader):
            img = img.cuda()

            # obtain prediction
            pred_joint, pred_rel = model(img)

            # convert prediction and label
            pred_joint = pred_joint.argmax(-1).data.cpu().numpy()
            pred_rel = pred_rel.data.cpu().numpy()
            pred_rel[pred_rel>=0.5] = 1
            pred_rel[pred_rel<0.5] = 0
            joint_label = joint_label.argmax(-1).data.numpy()
            rel_label = rel_label.data.numpy()

            # compute accuracy for joint categorization
            joint_acc.extend(pred_joint==joint_label)

            # compute True/False positive for relationship prediction
            rel_pos += np.sum(rel_label)
            rel_tp += np.sum(rel_label*pred_rel)
            rel_fp += np.sum((1-rel_label)*pred_rel)

    joint_acc = np.mean(np.array(joint_acc))
    precision = rel_tp/(rel_tp+rel_fp)
    recall = rel_tp/rel_pos
    f1_score = 2*(precision*recall)/(precision+recall)

    print('Test joint accuracy is %.3f' %joint_acc)
    print('Test relationship precision is %.3f' %precision)
    print('Test relationship recall is %.3f' %recall)
    print('Test relationship F1 score is %.3f' %f1_score)


if args.mode == 'train':
    main()
else:
    eval()
