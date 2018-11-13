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

pedestrian_file='result_faster_1class_0528.txt'
cyclist_file='result_cyclist_0525.txt'

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res152':('res152_faster_rcnn_iter_280000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',),'coco':('coco_2014_train+coco_2014_valminusminival',)}

def vis_detections(im, image_name, class_name, dets,map_file,thresh=0.4):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
   # save_path='result_image_new'
   # im = im[:, :, (2, 1, 0)]
    image_name=image_name.split('/')[-1].split('.')[0]
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        xmin=int(bbox[0])
        ymin=int(bbox[1])
        xmax=int(bbox[2])
        ymax=int(bbox[3])
        map_file.write("{image_id} {confidence} {xmin} {ymin} {xmax} {ymax}\n".format(image_id=image_name.replace('.jpg',''),confidence=score,xmin=xmin,ymin=ymin,xmax=xmax,ymax=ymax))
   #     cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),1)
   # cv2.imwrite(os.path.join(save_path,image_name),im)     
   # return im

def demo(sess, net, image_name,imagedir, mode,pedestrian,cyclist):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(imagedir, image_name)
    im = cv2.imread(im_file)
    save_path='result_image_new'
    # Detect all object classes and regress object bounds
    if mode == 'fast':
        scores, boxes = im_detect_fast(sess, net, im)
    else:
        multi_scales=[600,800,1000]    
        scores, boxes = im_detect(sess, net, im,multi_scales)
    # Visualize detections for each class
    CONF_THRESH = 0.4

    # Visualize people
    for cls_ind,cls in enumerate(CLASSES[1:]):
        cls_ind += 1 
   # cls = 'pedestrian'
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep=soft_nms(dets,sigma=0.6,Nt=0.3,method=2)
        dets=keep
    #    if cls=='pedestrian':
    #       map_file=pedestrian
    #    elif cls=='cyclist':
        map_file=pedestrian
        vis_detections(im, image_name, cls, dets,map_file,thresh=CONF_THRESH)
#    cv2.imwrite(os.path.join(save_path,image_name),im)
    

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res152')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
    parser.add_argument('--inputpath', dest='inputpath', help='image-directory', default="")
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
    inputpath=args.inputpath
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
    print(inputpath)
    print(inputlist)
    if len(inputpath) and inputpath != '/':
        for root,dirs,files in os.walk(inputpath):
            im_names=files
    elif len(inputlist):
        with open(inputlist,'r') as f:
            im_names = []
            for line in f.readlines():
                im_names.append(line.split('\n')[0])
    else:
        raise IOError('Error: ./run.sh must contain either --indir/--list')
#    map_file=open("result_finetune_0525.txt",'w')
    pedestrian=open(pedestrian_file,'w')
    cyclist=open(cyclist_file,'w')
    for im_name in tqdm(im_names):
        #print('Human detection for {}'.format(im_name))
       demo(sess, net, im_name,inputpath, mode,pedestrian,cyclist)
    pedestrian.close()
    cyclist.close()
