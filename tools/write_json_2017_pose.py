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
import json
from newnms.nms import  soft_nms
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
from tqdm import tqdm
import time
#from WIDER_Pedestrian_evaluation import fuck
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
import h5py

CLASSES=('__background__',
         'pedestrian')

batch=36046
soft_thresh=0.01
size=[700,800,900]
filepath='results/posetrack_val_batch{}_size{}_softthresh{}.txt'.format(batch,size,soft_thresh)
image_dir='/export/home/zby/faster-r-cnn-tf-all/data/VOCdevkit2007/VOC2007/JPEGImages'

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'res152':('res152_faster_rcnn_iter_{}.ckpt'.format(batch),)}
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
        map_file.write("{} {} {} {} {} {}\n".format(image_name, score,xmin,ymin,xmax,ymax))


def demo(sess, net, mode, im):
    """Detect object classes in an image using pre-computed object proposals."""
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
		#image_name=image_name.split('/')[-1].split('.')[0]
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes,
						  cls_scores[:, np.newaxis])).astype(np.float32)
		keep=soft_nms(dets,sigma=0.6,Nt=0.3,method=2)
		dets=keep
		#print(dets)
		inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
		bbox_list = []
		for i in inds:
			bbox = dets[i, :4]
			score = np.round(dets[i, -1]*1000)/1000
			xmin=round(bbox[0],1)
			ymin=round(bbox[1],1)
			xmax=round(bbox[2],1)
			ymax=round(bbox[3],1)
			bbox_list.append([xmin, ymin, xmax, ymax, score])
		return bbox_list
  #  cv2.imwrite(os.path.join(save_path,image_name+'.jpg'),im)
    
def data_preprocess(json_name, json_path):
	json_file = os.path.join(json_path, json_name)
	with open(json_file,'r') as f:
		annolist = json.load(f)['annolist']
	image_list = []
	for anno in annolist:
		image_name = anno['image'][0]['name']
		image_list.append(image_name)
	return image_list

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res152')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc')
#    parser.add_argument('--inputpath', dest='inputpath', help='image-directory', default="")
    parser.add_argument('--inputlist', dest='inputlist', help='image-list', default="test_posetrack.txt")
    parser.add_argument('--mode', dest='mode',help='detection mode, fast/normal/accurate', default="normal")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	args = parse_args()

	# model path
	demonet = args.demo_net
	dataset = args.dataset
	tfmodel = os.path.join('../output', 'res152', DATASETS[dataset][0], 'default',
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
						  tag='default',  anchor_scales=[8,16,32],anchor_ratios=[1,2.5,4])
	saver = tf.train.Saver()
	saver.restore(sess, tfmodel)

	print('Loaded network {:s}'.format(tfmodel))
	print(inputlist)

	json_path = '/export/home/zby/faster-r-cnn-tf-all/data/posetrack/test_2017_json'
	json_names = os.listdir(json_path)
	bbox_dict = dict()
	pbar = tqdm(range(len(json_names)))
	for json_name in json_names:
		pbar.update(1) 
		image_dict = dict()
		image_list = data_preprocess(json_name, json_path)
		for im_name in image_list:
			start = time.time()
			im_file = os.path.join('/export/home/zby/faster-r-cnn-tf-all/data/posetrack/test_2017_images',im_name)
			im = cv2.imread(im_file)
			bbox_list = demo(sess, net, mode, im)
			image_dict[im_name] = bbox_list
			pbar.set_description('Processing video {}: detection {}  takes about {:.2f}s'.format(json_name, im_name, time.time()-start))
		bbox_dict[json_name] = image_dict
	pbar.close()
	save_path = 'tf_detection_result_test_2017.json'
	with open(save_path,'w') as f:
		json.dump(bbox_dict,f)
