from __future__ import division
import time
import numpy as np
#from util import *
import os 
import os.path as osp
from voc_eval_dif import voc_eval

classes={"pedestrian"}
    
def eval(annotation_path="bq/Annotations",imageset_path="bq/ImageSets",result_path="result.txt",output_dir='output',iou_thresh=0.5,use_07_metric=True):
    annotation_path=os.path.join(annotation_path,'{:s}.xml')
    imageset_path=os.path.join(imageset_path,'Main','test.txt')
    aps=[]
    recs=[]
    pres=[]
    print("use 07_metric = {}".format(use_07_metric))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    aps = []
    for index in range(10):
      thresh = 0.5 + index*0.05
      for i,cls in enumerate(classes):
    #1 detection result should be a.txt
    #2 annotation filepath 
    #3 should be a image filepath
    #4 should be a kind of class like :"person"
        rec,prec,ap=voc_eval(result_path,annotation_path,imageset_path,cls,ovthresh=thresh,use_07_metric=use_07_metric)
        aps+=[ap]
        if thresh == 0.5:
          print('AP for thresh 0.5 = {:.4f}'.format(ap))
        if thresh == 0.75:
          print('AP for thresh 0.75 = {:.4f}'.format(ap))
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    print(aps)
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')



if __name__ ==  '__main__':
    eval(annotation_path="/export/home/zby/faster-r-cnn-tf-all/data/VOCdevkit2007/VOC2007/Annotations",
         imageset_path="/export/home/zby/faster-r-cnn-tf-all/data/VOCdevkit2007/VOC2007/ImageSets",
         result_path="results/posetrack_val_batch36046_size[700, 800, 900]_softthresh0.01.txt",
         output_dir='output',
         iou_thresh=0.5, 
         use_07_metric=False)
    
    
        
        
    
    
