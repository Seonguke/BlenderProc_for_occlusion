#%%
import os
import random

import cv2
import pyexr

import numpy as np

import pandas as pd 


from typing import Dict, Optional, List, Tuple


import matplotlib.pyplot as plt




folder_root = 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230302\\'

#folder_list = os.listdir(folder_root)
folder_list = ['002c110c-9bbc-4ab4-affa-4225fb127bad']


#print(folder_list)
print("folder files : ", len(folder_list))


print("start")


def dishow(disp):
    print("show")
    '''
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    plt.show()
    #plt.savefig(fname=workspace +'test.jpg', bbox_inches='tight', pad_inches=0)
    #plt.close()
    '''

i = 18

#workspace = './test230302/t2/'
workspace = './test230302/002c110c-9bbc-4ab4-affa-4225fb127bad/'


'''
panoptic_origin = cv2.imread(workspace + 'panoptic_origin_0016.png', 1)
#plt.imshow(panoptic_origin)
dishow(panoptic_origin)
'''


rm_count = 0


pixel_count = 320 * 240

new_object_mask = cv2.imread(workspace + 'mask_' + str(i+1).zfill(4)  + '.png', 0)
dishow(new_object_mask)
plt.close()

new_object_mask_1D = new_object_mask.reshape(-1)
new_object_mask_count = np.count_nonzero(new_object_mask_1D == 255)
print(new_object_mask_count)

percent_new_object_mask = (new_object_mask_count / pixel_count)
print(percent_new_object_mask)

if (percent_new_object_mask > 0.75):
    print("Too big percent_new_object_mask : ", percent_new_object_mask)
    print("x")
    rm_count = rm_count + 1
if (percent_new_object_mask < 0.03):
    print("Too small percent_new_object_mask : ", percent_new_object_mask)
    print("x")
    rm_count = rm_count + 1


segmap = np.load(workspace + 'segmap_' + str(i).zfill(4) + '.npy')
segmap0 = segmap[:,:,0]
#plt.imshow(segmap0)
dishow(segmap0)
plt.close()


segmap0_unique = np.unique(segmap0)
print(segmap0_unique)
print(len(segmap0_unique))

if (len(segmap0_unique) < 3):
    print('x')
    rm_count = rm_count + 1





#new_object_pre_unique = np.unique(new_object_mask)
#print(new_object_pre_unique)


#panoptic_origin = cv2.imread(workspace + 'panoptic_origin_0016.png')
panoptic_origin = cv2.imread(workspace + 'panoptic_origin_' + str(i).zfill(4) + '.png', 0)
#plt.imshow(panoptic_origin)
dishow(panoptic_origin)
plt.close()

panoptic_150 = panoptic_origin + 150
#plt.imshow(panoptic_150)
dishow(panoptic_150)
plt.close()



# 1이 벽, 2가 바닥
segmap_mask_2 = np.where(segmap0 == 2, 0, 1)
#plt.imshow(segmap_mask_2)
dishow(segmap_mask_2)
plt.close()


panoptic_150 = panoptic_150 * segmap_mask_2
#plt.imshow(panoptic_150)
dishow(panoptic_150)
plt.close()


panoptic_150 = np.where(panoptic_150 == 0, 2, panoptic_150)
dishow(panoptic_150)
plt.close()


segmap_mask_1 = np.where(segmap0 == 1, 0, 1)
#plt.imshow(segmap_mask_1)
dishow(segmap_mask_1)
plt.close()


panoptic_150 = panoptic_150 * segmap_mask_1
#plt.imshow(panoptic_150)
dishow(panoptic_150)
plt.close()


panoptic_150 = np.where(panoptic_150 == 0, 1, panoptic_150)
dishow(panoptic_150)
plt.close()



pre_unique = np.unique(panoptic_150)
print(pre_unique)
#for i in pre_unique:




print('----------------------------------------')

percent_80 = 0
percent_20 = 0



for i in pre_unique:
    mask = i == panoptic_150
    #print(mask)
    print("\n\n\nid : ",i)
    
    mask_1D = mask.reshape(-1)
    #mask_1D = mask.flatten()
    #print(mask_1D)
    #print(mask_1D.shape)



    
    if (i == 2):
        mask_occlusion = mask * new_object_mask
        mask_occlusion_1D = mask_occlusion.reshape(-1)
        count_occlusion = np.count_nonzero(mask_occlusion_1D == 255)
        print(count_occlusion)

        count_all = np.count_nonzero(mask_1D == True)
        print(count_all)

        percent = (count_occlusion / count_all)
        print(percent)
        if (percent >= 80):
            print("바닥 너무 많이 가림 : ", percent)
            print("x")
            rm_count = rm_count + 1

            
    if (i > 2):

        count_all = np.count_nonzero(mask_1D == True)
        print(count_all)
        

        dishow(mask)
        plt.close()


        if (count_all < 1000):
            mask = 1 - mask
            panoptic_150 = panoptic_150 * mask
            continue


        #mask는 boolean, mask_occlusion은 0과 255

        mask_occlusion = mask * new_object_mask
        
        dishow(mask_occlusion)
        plt.close()

        mask_occlusion_1D = mask_occlusion.reshape(-1)
        count_occlusion = np.count_nonzero(mask_occlusion_1D == 255)
        print(count_occlusion)

        percent = (count_occlusion / count_all)
        print(i , " percent : ", percent)

        if (percent >= 0.8):
            print("80퍼이상 겹침")
            percent_80 = percent_80 + 1       

        if (percent >= 0.2):
            print("20퍼 이상 겹침")
            percent_20 = percent_20 + 1







    


dishow(panoptic_150)
plt.close()


pre_unique = np.unique(panoptic_150)
print(pre_unique)
print(len(pre_unique))

if (len(pre_unique) < 2):
    print("물체가 없음")
    print("x")
    rm_count = rm_count + 1

print("ing")


print(percent_80)
print(percent_20)


print("---------------------")

if (percent_80 >= 1):
    print("80퍼 이상 가림 : ", percent_80)
    print("x")
    rm_count = rm_count + 1
if (percent_20 == 0):
    print("20퍼 이상 가림 : ", percent_20)
    print("x")
    rm_count = rm_count + 1



if (rm_count > 0):
    print('지워야 함 : ', rm_count)
    print("remove")


# %%
