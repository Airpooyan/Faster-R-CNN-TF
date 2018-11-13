import cv2
import os

f=open('test_image.txt','r')
lines=f.readlines()
outpath='images'
for line in lines:
  line=line.strip()
  image=cv2.imread(line)
  image_name=line.split('/')[-1]
  print(image_name)
  save_path=os.path.join(outpath,image_name)
  cv2.imwrite(save_path,image)
