import os
import cv2
import pyexr
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

#root= 'D:\\my_test_uk\\'
root= 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\data230306_filtering\\'
file_list = os.listdir(root)
f= open(root + './data.txt','w')
for dir in file_list:
    for i in range(20):
        if i%2 > 0:
            continue
        try :
            name=root+dir + '\\' + 'depth_' + str(i).zfill(4) + '.exr'
            img = pyexr.read(name)

        except:
            #os.remove(root+dir)
            try:
                shutil.rmtree(root+dir)
            except:
                continue
            continue
        data = dir + '\\' + 'depth_' + str(i).zfill(4) + '.exr'
        f.write(data + '\n')

def bounding_box_mask(mask):
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    return x, y, w, h

f.close()
f = open(root + './data.txt','r')
data=[]
lines = f.readlines()
cnt= 0
for line in lines:
    line = line.strip()
    depth = pyexr.read(root+line)
    jsn = line.split('\\')
    nums = line.split('_')
    num= nums[1].split('.')[0]
    depth2 = pyexr.read(root + jsn[0]+'\\'+'depth_'+str(int(num)+1).zfill(4)+'.exr')
    #print(num)



    depth = depth[:,:,0]
    depth2 = depth2[:,:,0]
    mask = depth2 - depth
    mask[mask!=0]=255
    cv2.imwrite(root + jsn[0] + '\\' + 'mask_' + str(int(num) + 1).zfill(4) + '.png', mask)
    try:
        x,y,w,h=bounding_box_mask(mask)

        bb_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        bb_mask[y:y+h, x:x+w] = 255
        cv2.imwrite(root+jsn[0]+'\\'+'bb_mask_'+str(int(num)+1).zfill(4)+'.png',bb_mask)
        data.append(line)
    except:
        continue

#print(file_list)

train_names, test_names = train_test_split(data, test_size=0.1, random_state=42)
train_names, val_names = train_test_split(train_names, test_size=0.1, random_state=42)
f=open(root + './train.txt','w')
for line in train_names:
    line = line.strip()
    f.write(line + '\n')
f=open(root + './val.txt','w')
for line in val_names:
    line = line.strip()
    f.write(line + '\n')
f=open(root + './test.txt','w')
for line in test_names:
    line = line.strip()
    f.write(line + '\n')


f.close()
