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

#file_list = ['00ad8345-45e0-45b3-867d-4a3c88c2517a']

num = 0
for dir in file_list:

    print("\n ---------------------------------------------- \n")
    print(dir)
    print(num ," : ", len(file_list))
    num = num + 1

    if num < 638:
        continue

    workspace = root + dir + '\\'

    img_list = os.listdir(workspace)
    img_list = [file for file in img_list if file.startswith("rgb")]
    
    if (len(img_list) == 0):
        continue
    
    print(img_list)

    

    fig = plt.figure(figsize = (20, 16))
    rows = 5
    cols = 4
    for k in range(0, len(img_list)):
        img = cv2.imread(workspace + img_list[k])
        ax = fig.add_subplot(rows, cols, k+1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(img_list[k])
        ax.axis("off")

    plt.show()

    input_list = list(input().split())
    input_list = list(map(int, input_list))
    print(input_list)

    rm_list = []
    for i in range(0, len(input_list)):
        img_name = 'rgb_' + str(input_list[i]).zfill(4)  + '.png'
        depth_name = 'depth_' + str(input_list[i]).zfill(4)  + '.exr'
        rm_list.append(img_name)
        rm_list.append(depth_name)

    print(rm_list)

    for e in rm_list:
        #print(e)

        x_img_name = workspace + '\\' + 'garbage_' + e

        e =  workspace + '\\' + e
        #print(e)
        #os.remove(e)
        os.rename(e, x_img_name)




    new_img_list = os.listdir(workspace)
    new_img_list = [file for file in new_img_list if file.startswith("rgb")]

    fig = plt.figure(figsize = (20, 16))
    rows = 5
    cols = 4
    for k in range(0, len(new_img_list)):
        img = cv2.imread(workspace + new_img_list[k])
        ax = fig.add_subplot(rows, cols, k+1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(new_img_list[k])
        ax.axis("off")

    plt.show()
        






# %%
