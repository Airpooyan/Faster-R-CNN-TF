#!/usr/bin/env python
 
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect, im_detect_fast
#from model.nms_wrapper import nms
from newnms.nms import  soft_nms
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
from tqdm import tqdm

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import h5py

CLASSES=('__background__',
         'pedestrian')

batch=67825
soft_thresh=0.5
size=[1500,1800]
filepath='results/faster_res152_batch{}_size{}_{}softthresh{}.txt'.format(batch,size[0],size[1],soft_thresh)
imagedir='/export/home/zby/faster-r-cnn-tf-1class/data/VOCdevkit2007/VOC2007/JPEGImages'

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res152':('res152_faster_rcnn_iter_67825.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'coco':('coco_2014_train+coco_2014_valminusminival',)}

def vis_detections(im, image_name, class_name, dets,map_file,thresh=0.4):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
   # save_path='result_image_new'
   # im = im[:, :, (2, 1, 0)]
   # image_name=image_name.split('/')[-1].split('.')[0]
    for i in inds:
        bbox = dets[i, :4]
        score = np.round(dets[i, -1]*1000)/1000
        xmin=round(bbox[0],1)
        ymin=round(bbox[1],1)
        xmax=round(bbox[2],1)
        ymax=round(bbox[3],1)
        #image_id score xmin ymin w h
        map_file.write("{} {} {} {} {} {}\n".format(image_name+'.jpg',score,xmin,ymin,xmax-xmin,ymax-ymin))
   #     cv2.rectangle(im,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,0,255),1)
   # cv2.imwrite(os.path.join(save_path,image_name),im)     
   # return im

def demo(sess, net, image_name,imagedir, mode,map_file):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(imagedir, image_name)
    im = cv2.imread(im_file)
    save_path='result_image_new'
    # Detect all object classes and regress object bounds
    if mode == 'fast':
        scores, boxes = im_detect_fast(sess, net, im)
    else:
        multi_scales=size
        scores, boxes = im_detect(sess, net, im,multi_scales)
    # Visualize detections for each class
    CONF_THRESH = soft_thresh

    # Visualize people
    for cls_ind,cls in enumerate(CLASSES[1:]):
        cls_ind += 1 
   # cls = 'pedestrian'
        image_name=image_name.split('/')[-1].split('.')[0]
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep=soft_nms(dets,sigma=0.6,Nt=0.4,method=2)
        dets=keep
        #print(dets)
        vis_detections(im, image_name, cls, dets,map_file,thresh=CONF_THRESH)
  #  cv2.imwrite(os.path.join(save_path,image_name+'.jpg'),im)
    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res152')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
#    parser.add_argument('--inputpath', dest='inputpath', help='image-directory', default="")
    parser.add_argument('--inputlist', dest='inputlist', help='image-list', default="")
    parser.add_argument('--mode', dest='mode',help='detection mode, fast/normal/accurate', default="normal")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('../output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    inputlist = args.inputlist
    mode = args.mode
 
    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet =='res152':
        net = resnetv1(num_layers=152)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 2,
                          tag='default', anchor_scales=[2,4,8, 16, 32],anchor_ratios=[0.5,1,2,2.7])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    im_names = []
    print(inputlist)
    with open(inputlist,'r') as f:
         for line in f.readlines():
             im_names.append(line.split('\n')[0]+'.jpg')
    map_file=open(filepath,'w')
  
    for im_name in tqdm(im_names):
        #print('Human detection for {}'.format(im_name))
       demo(sess, net, im_name,imagedir, mode,map_file)
    map_file.close()
