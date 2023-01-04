import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataloader import LSP_joint_generator
from model import PoseNet, PoseNet_slim, ResNet50
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
parser.add_argument('--epoch', type=int, default=60, help='Defining maximal number of epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='Defining initial learning rate (default: 1e-3)')
parser.add_argument('--batch_size', type=int, default=128, help='Defining batch size for training (default: 32)')
parser.add_argument('--clip', type=float, default=0, help='Gradient clipping to prevent gradient explode (default: 0)')
parser.add_argument('--joint_size', type=int, default=32, help='Defining size of the patches for joints')
parser.add_argument('--num_cluster', type=int, default=11, help='Defining number of clusters for pairwise relationship')
parser.add_argument('--alpha', type=float, default=1, help='balancing factor for training')
parser.add_argument('--use_rel', type=bool, default=False, help='using clustering relationship or not for ResNet')

args = parser.parse_args()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_learning_rate(init_lr,optimizer, epoch):
    "adatively adjust lr based on epoch"
    lr = init_lr * (0.25 ** int((epoch+1)/20)) #previously 0.25/40

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    tf_summary_writer = tf.summary.create_file_writer(args.checkpoint_dir)    
    
    # define dataloader
    train_data = LSP_joint_generator(args.img_dir, args.anno_dir, 'train', args.joint_size)
    val_data = LSP_joint_generator(args.img_dir, args.anno_dir, 'val', args.joint_size)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, 
                                    shuffle=True, num_workers=12)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, 
                                    shuffle=False, num_workers=4)

    # initialize model
    # model = PoseNet(args.num_cluster)
    # model = PoseNet_slim(args.num_cluster)
    if not args.use_rel:
        model = ResNet50()
    else:
        model = ResNet50(args.use_rel, train_data.cluster_cat,
                    train_data.cluster2joint)
    model = nn.DataParallel(model)
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

        for batch_idx,(img, joint_label, rel_label, data_id) in enumerate(trainloader):
            img, joint_label, rel_label = img.cuda(), joint_label.cuda(), rel_label.cuda()
            optimizer.zero_grad() 

            if not args.use_rel:
                pred_joint = model(img)
                joint_loss = cross_entropy(pred_joint, joint_label)
                loss = joint_loss
            else:
                pred_joint, pred_rel = model(img)

                joint_loss = cross_entropy(pred_joint, joint_label)
                rel_loss = cross_entropy(pred_rel, rel_label)
                loss = joint_loss + args.alpha*rel_loss

            loss.backward()

            if not args.clip == 0 :
                clip_grad_norm_(model.parameters(),args.clip)
            optimizer.step()

            avg_joint_loss = (avg_joint_loss*np.maximum(0, batch_idx) + 
                        joint_loss.data.cpu().numpy())/(batch_idx+1)
            if args.use_rel:
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
            for batch_idx,(img, joint_label, rel_label, data_id) in enumerate(valloader):
                img = img.cuda()

                # obtain prediction
                if not args.use_rel:
                    pred_joint = model(img)
                    pred_joint = pred_joint.argmax(-1).data.cpu().numpy()

                else:
                    pred_joint, pred_rel = model(img)
                    pred_joint = pred_joint.argmax(-1).data.cpu().numpy()
                    pred_rel = pred_rel.argmax(-1).data.cpu().numpy()
                
                # convert prediction and label
                joint_label = joint_label.argmax(-1).data.numpy()
                rel_label = rel_label.argmax(-1).data.numpy()

                # compute accuracy for joint categorization
                joint_acc.extend(pred_joint==joint_label)

                # compute accuracy for relationship categorization
                if args.use_rel:
                    rel_acc.extend(pred_rel==rel_label)

        joint_acc = np.mean(np.array(joint_acc))
        if args.use_rel:
            rel_acc = np.mean(np.array(rel_acc))

        with tf_summary_writer.as_default():
            tf.summary.scalar('Joint-Accuracy', joint_acc, step=iteration)
            if args.use_rel:
                tf.summary.scalar('Relation-Accuracy', rel_acc, step=iteration)

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
            torch.save(model.module.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
            # torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model_best.pth'))
            val_acc = cur_score
        torch.save(model.module.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))
        # torch.save(model.state_dict(),os.path.join(args.checkpoint_dir,'model.pth'))


def eval():
    # define dataloader
    test_data = LSP_joint_generator(args.img_dir, args.anno_dir, 'test', args.joint_size)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, 
                                    shuffle=False, num_workers=4)

    # initialize model
    # # model = PoseNet(args.num_cluster)
    # model = PoseNet_slim(args.num_cluster)
    if not args.use_rel:
        model = ResNet50()
    else:
        model = ResNet50(args.use_rel, test_data.cluster_cat,
                    test_data.cluster2joint)

    model.load_state_dict(torch.load(args.weights))
    model = model.cuda()
    model.eval()

    # Evaluate the performance for both joint categorization and relationship prediction
    joint_acc = []
    rel_tp = 0
    rel_fp = 0
    rel_pos = 0
    # record the joints that can or can not be easily classified in each image
    record_acc = dict()

    with torch.no_grad():
        for batch_idx,(img, joint_label, rel_label, data_id) in enumerate(testloader):
            img = img.cuda()

            # obtain prediction
            if args.use_rel:
                pred_joint, pred_rel = model(img)
            else:
                pred_joint = model(img)
            
            # compute the predicted confidence on the correct classes
            pred_joint = pred_joint.data.cpu().numpy()
            joint_label = joint_label.data.numpy()
            pred_confidence = (pred_joint*joint_label).sum(-1)
            
            for idx in range(len(data_id)):
                cur_id = data_id[idx]
                img_id, joint_id = cur_id.split('_')
                if img_id not in record_acc:
                    record_acc[img_id] = dict()
                    record_acc[img_id]['high'] = []
                    record_acc[img_id]['low'] = []
                if pred_confidence[idx]<0.5: 
                    record_acc[img_id]['low'].append(joint_id)
                if pred_confidence[idx]>0.8:
                    record_acc[img_id]['high'].append(joint_id)

            # convert prediction and label for accuracy computation
            pred_joint = pred_joint.argmax(-1)
            joint_label = joint_label.argmax(-1)

            # compute accuracy for joint categorization
            joint_acc.extend(pred_joint==joint_label)

    joint_acc = np.mean(np.array(joint_acc))

    print('Test joint accuracy is %.3f' %joint_acc)   

    with open('pred_confidence_36.json', 'w') as f:
        json.dump(record_acc, f)

if args.mode == 'train':
    main()
else:
    eval()
