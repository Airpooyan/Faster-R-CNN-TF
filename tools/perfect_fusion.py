#import sys
#sys.path.insert(0,'/home/cp/py-faster-rcnn/tools')
#import _init_paths
#from fast_rcnn.nms_wrapper import nms
import numpy as np
#import caffe
import copy
import time
#caffe.set_mode_gpu()
#caffe.set_device(4)
#a = open('/home/zyf/caffe/py-faster-rcnn/round2_faster_multi_rpn_context_688_result.csv','r')
#a = open('/home/cp/R-FCN-test/round2_rfcn_context_600_result.csv','r')
#a = open('../round2_rfcn_context_fusion.csv','r')
a = open('581.txt','r')
#a = open('fusion_rfcn_2faster.csv','r')
#a = open('/home/cp/RFCN-group/R-FCN-sign/round2_sign_600_result.csv','r')
a = a.readlines()
#b = open('/home/zyf/caffe/py-faster-rcnn/round2_faster_multi_rpn_context_max.csv','r')
#b = open('4_light.csv','r')
b = open('582.txt','r')
#b = open('/home/cp/R-FCN-test/round2_rfcn_context_800_result.csv','r')
#b = open('/home/zyf/caffe/py-faster-rcnn/round2_faster_multi_rpn_context_max.csv','r')
#b = open('/home/cp/RFCN-group/R-FCN-sign/round2_sign_800_result.csv','r')
b = b.readlines()
#c = open('/home/zyf/caffe/py-faster-rcnn/round2_faster_multi_rpn_context_1000_result.csv','r')
#c = open('4_sign.csv','r')
#c = open('/home/cp/py-R-FCN/round2_rfcn_1000_result.csv','r')
#c = c.readlines()
#d = open('/home/cp/RFCN-group/R-FCN-sign/round2_sign_1200_result.csv','r')
#d = open('/home/cp/R-FCN-test/round2_rfcn_context_1200_result.csv','r')
#d = open('/home/cp/py-R-FCN/round2_rfcn_1200_result.csv','r')
#d = d.readlines()

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    #print "order:",order

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        chonghe = np.where(ovr > thresh)[0]
        for j in chonghe:
           scores[i] += scores[order[j+1]]
        
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def to_dict(a,weight=1):
	ns = dict()
	for i in range(0,len(a)):
		j = a[i].strip()
		#print(j)
		img,score,x1,y1,w,h = j.split(' ')
		x2 = float(x1)+float(w)
		y2 = float(y1)+float(h)
		score = float(score) * weight
		#img = int(img)
		b = np.array([x1,y1,x2,y2,score]).astype(np.float32).reshape(1,5)
		if img in ns:
			ns[img] = np.vstack((ns[img],b))
		else:
			ns[img] = b
	return ns

def fusion_dict(a,b):
	c = copy.deepcopy(b)
	for key in a:
		if key in c:
			c[key] = np.vstack((a[key],c[key]))
		else:
			c[key] = a[key]
	return c

def dict_nms(ns,NMS_THRESH):
	for key in ns:
		dets = ns[key]
		keep = nms(dets,NMS_THRESH)
		dets = dets[keep,:]
		ns[key] = sorted(dets,key = lambda i:i[4],reverse = True )
	return ns
'''
i = {4:[[0,1]],5:[[3,4]]}
j = {2:[[3,4]],3:[[5,6]]}
k = fusion_dict(i,j)
for key in k:
	for i in k[key]:
		x1,y1 = i
		print x1,y1
'''			
t0 = time.time()
a_dict = to_dict(a,0.3)
b_dict = to_dict(b,0.7)
#print(a_dict)
#c_dict = to_dict(c)
#d_dict = to_dict(d)
nab = fusion_dict(a_dict,b_dict)
#ncd = fusion_dict(c_dict,d_dict)
#test = d_dict
#ns = fusion_dict(nab,ncd)
#ns = fusion_dict(nab,c_dict)
#nab = dict_nms(nab,0.3)
ns = dict_nms(nab,0.9)
e = open('fusion.txt','w')
#e.write('id,x1,y1,x2,y2,score\n')
count = 0
for key in ns:
	img = str(key)
	lines = ns[key]
	for i in lines:
		x1,y1,x2,y2,score = i
		w = x2-x1
		h = y2-y1
		'''
		score = round(float(score),2)
		if score > 1.0:
			score = 1.0
			count += 1
		if score == 0.0:
			score = 0.01
		if score > 0.02:
			line = ' '.join([str(img),str(score),str(x1),str(y1),str(w),str(h)])+'\n'
		'''
		#if score > 1.0:
		#	score = 1.0
		line = ' '.join([str(img),str(score),str(x1),str(y1),str(w),str(h)])+'\n'
		#print line,'is writed'
		e.write(line)
print ('cost ',time.time()-t0,'s')
print ('count:',count)
