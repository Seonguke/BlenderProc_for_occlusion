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

import csv

from collections import Counter

print("start")


def dishow(disp):
    #print("show")
    
    '''
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    plt.show()
    #plt.savefig(fname=workspace + 'show/' + 'test.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()
    '''
    


def mask_show(panoptic_150, pre_unique):

    #print("---------------- showing objects ----------------")


    dishow(panoptic_150)

    for i in pre_unique:
            mask = i == panoptic_150
            #print(mask)
            #print("\n\n\nid : ",i)
            
            mask_1D = mask.reshape(-1)
            count_all = np.count_nonzero(mask_1D == True)
            #print("point 개수 : " ,count_all)


            #mask_1D = mask.flatten()
            #print(mask_1D)
            #print(mask_1D.shape)

            dishow(mask)


def count_filter(panoptic_150, pre_unique):

    #dishow(panoptic_150)

    print("--------- count filtering --------------")
 

    for i in pre_unique:
            
            # ----------- 바닥 제외 추가
            if (i == min(pre_unique)):
                continue

            mask = i == panoptic_150
            #print(mask)
            print("\n\n\nid : ",i)
            
            mask_1D = mask.reshape(-1)
         
            dishow(mask)
            count_all = np.count_nonzero(mask_1D == True)
            print("point 개수 : " ,count_all)
            
            if (count_all < 1000):
                dishow(mask)
                mask = 1 - mask
                panoptic_150 = panoptic_150 * mask
                print("ingnore mask")
    
    return panoptic_150


def get_mask(pre_mask):
    pre_mask = []
    pre_unique = np.unique(pre[..., 0])
    #print(pre_unique)
    for i in pre_unique:

        mask = i == pre[..., 0]
        if i==1 or i==2: # i not in 중복
            pre_mask.append(mask.astype(bool))
            continue
        h, w = mask.shape
        max_area = (h * w) - mask.sum()
        mask = mask.astype(np.uint8) * 255
        ret, markers = cv2.connectedComponents(mask)
        markers = markers + 1
        unique = np.unique(markers)

        #print(unique)


        for j in unique:
            small_mask = markers == j
            if small_mask.sum() < 200 or small_mask.sum() >= max_area:
                continue
            pre_mask.append(small_mask)
    return pre_mask

def get_mask_LDY(pre_mask, occlusion_num):
    pre_mask = []
    pre_unique = np.unique(pre[..., 0])
    #print(pre_unique)
    #print(occlusion_num)
    for i in pre_unique:
        #print(i)
        mask = i == pre[..., 0]
        if (i==1 or i==2): # i not in 중복
            pre_mask.append(mask.astype(bool))
            continue

        if (i not in occlusion_num): # i not in occlusion
            continue

        h, w = mask.shape
        max_area = (h * w) - mask.sum()
        mask = mask.astype(np.uint8) * 255
        ret, markers = cv2.connectedComponents(mask) #인접 픽셀 기반 레이블링
        markers = markers + 1
        unique = np.unique(markers)


        for j in unique:
            small_mask = markers == j
            if small_mask.sum() < 200 or small_mask.sum() >= max_area:
                continue
            pre_mask.append(small_mask)
    return pre_mask




root= 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\data230306_filtering\\'
file_list = os.listdir(root)

file_list = [file for file in file_list if not file.endswith(".txt")]

print(file_list)
print(len(file_list))

#file_list = ['002c110c-9bbc-4ab4-affa-4225fb127bad']

num = 0
for dir in file_list:


    #json file 생성
    json_list = []
    print(dir)
    print(num ," : ", len(file_list))
    num = num + 1
    workspace = root + dir + '\\'



    for k in range(0, 20):

        #k = 6

        #print("------------- start -----------------------")
        #print(workspace)
        #print(k)

        index = '' + str(k) + ''
        index_inform = []
        index_inform.append(index)

        if k%2 > 0:
            #index_inform.append("Pass")
            index_inform.insert(0, "Pass")
            json_list.append(index_inform)
            continue



        '''
        segmap_now = np.load(workspace + 'segmap_' + str(k).zfill(4) + '.npy')
        segmap_now_0_unique = np.unique(segmap_now[..., 0])
        segmap_now_0_unique = list(segmap_now_0_unique)
        print(segmap_now_0_unique)


        segmap_next = np.load(workspace + 'segmap_' + str(k+1).zfill(4) + '.npy')
        segmap_next_0_unique = np.unique(segmap_next[..., 0])
        segmap_next_0_unique = list(segmap_next_0_unique)
        print(segmap_next_0_unique)
        print("-------------------------")
        '''


        rm_count = 0


        pixel_count = 320 * 240

        #print(workspace + 'mask_' + str(k+1).zfill(4)  + '.png', 0)
        new_object_mask = cv2.imread(workspace + 'mask_' + str(k+1).zfill(4)  + '.png', 0)
        dishow(new_object_mask)


        new_object_mask_1D = new_object_mask.reshape(-1)
        new_object_mask_count = np.count_nonzero(new_object_mask_1D == 255)
        #print(new_object_mask_count)

        percent_new_object_mask = (new_object_mask_count / pixel_count)
        #print(percent_new_object_mask)

        '''
        new_object_mask_TF = new_object_mask_1D.astype(bool)
        dishow(new_object_mask_TF)
        new_object_mask_TF_count = new_object_mask_TF.sum()
        percent_new_object_mask = (new_object_mask_TF_count / pixel_count)
        print(percent_new_object_mask)
        '''


        if (percent_new_object_mask > 0.75):
            #print("a : ", percent_new_object_mask)
            #print("x")
            rm_count = rm_count + 1
            rm_inform = "마스크 너무 큼 Too big percent_new_object_mask : " + str(percent_new_object_mask)
            index_inform.append(rm_inform)




        if (percent_new_object_mask < 0.03):
            #print("마스크 너무 작음 Too small percent_new_object_mask : ", percent_new_object_mask)
            #print("x")
            rm_count = rm_count + 1
            rm_inform = "마스크 너무 작음 Too small percent_new_object_mask : " + str(percent_new_object_mask)
            index_inform.append(rm_inform)


        pre = np.load(workspace + 'segmap_' + str(k).zfill(4) + '.npy')
        dishow(pre[:,:,0])
        '''
        pre_mask = get_mask_LDY(pre, overlap_cat_id)
        for mask in pre_mask:
            dishow(mask)
        '''

        segmap = np.load(workspace + 'segmap_' + str(k).zfill(4) + '.npy')
        segmap0 = segmap[:,:,0]
        #plt.imshow(segmap0)
        dishow(segmap0)



        segmap0_unique = np.unique(segmap0)
        #print(segmap0_unique)
        #print(len(segmap0_unique))

        if (len(segmap0_unique) < 3):
            #print("물체가 없음")
            #print("x")
            rm_count = rm_count + 1
            rm_inform = "물체가 없음 : " + str(segmap0_unique)
            index_inform.append(rm_inform)


    

        new_object_mask_10 = (new_object_mask/255).astype(int)
        dishow(new_object_mask_10)

        segmap0_checking = segmap0 
        segmap0_checking = segmap0_checking * new_object_mask_10
        dishow(segmap0_checking)
        occlusion_id = np.unique(segmap0_checking)
        #print(occlusion_id)

        '''
        occlusion_objects = []
        for h in range(0, len(occlusion_id)):
            if (occlusion_id[h] == 0 or occlusion_id[h] == 1 or occlusion_id[h] == 2):
                continue
            occlusion_objects.append
        '''

        occlusion_id = np.delete(occlusion_id, np.where((occlusion_id == 0) | (occlusion_id == 1) | (occlusion_id == 2)))
        #print(occlusion_id)

        #pre_mask = get_mask(segmap0)
        pre_mask = get_mask_LDY(segmap0, occlusion_id)
        for mask in pre_mask:
            dishow(mask)


        #print('----------------------------------------')

        percent_60 = 0
        percent_15 = 0


        for i in range(0, len(pre_mask)):
            mask = pre_mask[i]
            #print("\n\n\nid : ",i)


            if (i == 0):
                continue
            elif (i == 1):
                dishow(mask)
                #print(mask.sum())
                occlusion_area = mask * new_object_mask_10
                dishow(occlusion_area)
                #print(occlusion_area.sum())
                percent_occlusion = occlusion_area.sum()/mask.sum()
                #print("바닥 겹치는 % : ", percent_occlusion)
                
                if (percent_occlusion >= 0.8):
                    #print("바닥 너무 많이 가림 : ", percent_occlusion)
                    #print("x")
                    rm_count = rm_count + 1
                    rm_inform = "바닥 너무 많이 가림 : " + str(percent_occlusion)
                    index_inform.append(rm_inform)



            else:
                dishow(mask)
                #print(mask.sum())
                occlusion_area = mask * new_object_mask_10
                dishow(occlusion_area)
                #print(occlusion_area.sum())
                percent_occlusion = occlusion_area.sum()/mask.sum()
                #print("objcet 겹치는 % : ", percent_occlusion)

                if (percent_occlusion >= 0.6):
                    #print("60퍼이상 겹침")
                    percent_60 = percent_60 + 1       

                '''
                if (percent_occlusion >= 0.15):
                    print("15퍼 이상 겹침")
                    percent_15 = percent_15 + 1
                '''

                if (occlusion_area.sum() >= 1000):
                    #print("1000 pixel 이상 겹침")
                    percent_15 = percent_15 + 1





        

        #print("ing")


        #print(percent_60)
        #print(percent_15)


        #print("---------------------")

        if (percent_60 >= 1):
            #print("60퍼 이상 가림 : ", percent_60)
            #print("x")
            rm_count = rm_count + 1
            rm_inform = "60퍼 이상 가림 :" + str(percent_60)
            index_inform.append(rm_inform)
        if (percent_15 == 0):
            #print("1000 pixel 이상 가림 : ", percent_15)
            #print("x")
            rm_count = rm_count + 1
            rm_inform = "1000 pixel 이상 가림 : " + str(percent_15)
            index_inform.append(rm_inform)



        if (rm_count > 0):
            #print('지워야 함 : ', rm_count)
            #print("remove")
            rm_inform = "지워야함 : " + str(rm_count)
            index_inform.append(rm_inform)

        
        '''
        rm_inform = "rm count : " + str(rm_count)
        index_inform.append(rm_inform)
        '''


        if rm_count > 0:
            index_inform.insert(0, "False")
        else:
            index_inform.insert(0, "True")




        json_list.append(index_inform)

        #print(json_list)
    
    #print(json_list)



    #json_name = os.path.basename(dir)
    #print(workspace)
    json_path = workspace + dir  +"_rmlist" + ".json"
    '''
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent="\t")
    '''
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, indent="\t", ensure_ascii=False)



# %%
