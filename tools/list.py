import os

path='/export/home/zby/bq_testdata/val_data'
images=os.listdir(path)
f=open('val_data_list.txt','w')
for image in images:
    line=os.path.join(path,image)
    f.write(line+'\n')
f.close
