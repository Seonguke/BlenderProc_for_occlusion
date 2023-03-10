import os
import cv2
import pyexr
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

#root= 'D:\\my_test_uk\\'
root= 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230223_hand\\'
file_list = os.listdir(root)


all_max = 0

for dir in file_list:
    for i in range(10):
        
        
        rgb = root+dir + '\\' + 'rgb_' + str(i).zfill(4) + '.png'
        depth = root+dir + '\\' + 'depth_' + str(i).zfill(4) + '.exr'


        ''''
        #depth만 있고 rgb 없는거 삭제
        print(rgb)
        if(os.path.isfile(rgb)):
            print("file exist")
        else:
            print("X")
            if(os.path.isfile(depth)):
                os.remove(depth)
        '''


        
        #depth값 확인
        #print(depth)
        if(os.path.isfile(depth)):
            #print("file exist")
            depth = pyexr.read(depth)
            max_value = np.max(depth)
            print(max_value)
            if(max_value > all_max):
                all_max = max_value
                #print(all_max)
        #else:
            #print("X")
        

        '''
        #depth값 이상한거 변경
        if(os.path.isfile(depth)):
            #print("file exist")
            depth_value = pyexr.read(depth)
            max_value = np.max(depth_value)
            print(max_value)
            if(max_value > 10.0):
                print("outlier")
                depth_value_processing = np.where(depth_value > 10, 10, depth_value)
                pyexr.write(depth, depth_value_processing)
        '''
        

print(all_max)

