#%%
import os
import random

import cv2
import pyexr

import numpy as np

import pandas as pd 


from typing import Dict, Optional, List, Tuple


import matplotlib.pyplot as plt

import json




print("start")


def dishow(disp):
    print("show")
    
    
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    plt.show()
    #plt.savefig(fname=workspace + 'show/' + 'test.jpg', bbox_inches='tight', pad_inches=0)
    #plt.close()
    
    
    

#i = 18

#workspace = './test230302/t2/'
#workspace = './test230302/002c110c-9bbc-4ab4-affa-4225fb127bad/'


'''
panoptic_origin = cv2.imread(workspace + 'panoptic_origin_0016.png', 1)
#plt.imshow(panoptic_origin)
dishow(panoptic_origin)
'''


root= 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\data230306_filtering\\'
file_list = os.listdir(root)

file_list = [file for file in file_list if not file.endswith(".txt")]

print(file_list)

#file_list = ['t1']

num = 0
for dir in file_list:

    print("----------------------------------------------")
    print(dir)
    print(num ," : ", len(file_list))
    num = num + 1

    if num >= 638:
        break

    workspace = root + dir + '\\'

    
    img_list = os.listdir(workspace)
    img_list = [file for file in img_list if file.startswith("rgb")]
    
    if (len(img_list) == 0):
        continue
    
    #print(img_list)

    for i in range(len(img_list)):
        #name = img_list[i]
        img_list[i] = img_list[i].replace('rgb_', '')
        img_list[i] = img_list[i].replace('.png', '')

    
    #print(img_list)

    img_list = list(map(int, img_list))

    #print(img_list)
    
    for m in (img_list):
        #print(m)

        if m%2 > 0:
            n = m -1
        else:
            n = m + 1

        gar_name = workspace + '\\' + 'garbage_rgb_' + str(n).zfill(4) + '.png'
        #print(gar_name)
        if (os.path.isfile(gar_name)):
            print("changing")
            src_name = workspace + '\\' + 'rgb_' + str(m).zfill(4) + '.png'
            dst_name = workspace + '\\' + 'garbage_rgb_' + str(m).zfill(4) + '.png'
            os.rename(src_name, dst_name)

            src_name = workspace + '\\' + 'depth_' + str(m).zfill(4) + '.exr'
            dst_name = workspace + '\\' + 'garbage_depth_' + str(m).zfill(4) + '.exr'
            os.rename(src_name, dst_name)









# %%
