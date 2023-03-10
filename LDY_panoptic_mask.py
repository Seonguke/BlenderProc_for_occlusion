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
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    plt.show()
    #plt.savefig(fname=workspace +'test.jpg', bbox_inches='tight', pad_inches=0)
    #plt.close()



workspace = './test230302/t2/'


'''
panoptic_origin = cv2.imread(workspace + 'panoptic_origin_0016.png', 1)
#plt.imshow(panoptic_origin)
dishow(panoptic_origin)
'''



#panoptic_origin = cv2.imread(workspace + 'panoptic_origin_0016.png')
panoptic_origin = cv2.imread(workspace + 'panoptic_origin_0016.png', 0)
#plt.imshow(panoptic_origin)
dishow(panoptic_origin)
plt.close()

panoptic_150 = panoptic_origin + 150
#plt.imshow(panoptic_150)
dishow(panoptic_150)
plt.close()

segmap = np.load(workspace + 'segmap_0016.npy')
segmap0 = segmap[:,:,0]
#plt.imshow(segmap0)
dishow(segmap0)
plt.close()

# 1이 벽, 2가 바닥
segmap_mask_1 = np.where(segmap0 == 1, 0, 1)
#plt.imshow(segmap_mask_1)
dishow(segmap_mask_1)
plt.close()

segmap_mask_2 = np.where(segmap0 == 2, 0, 1)
#plt.imshow(segmap_mask_2)
dishow(segmap_mask_2)
plt.close()

panoptic_150 = panoptic_150 * segmap_mask_1
#plt.imshow(panoptic_150)
dishow(panoptic_150)
plt.close()

panoptic_150 = panoptic_150 * segmap_mask_2
#plt.imshow(panoptic_150)
dishow(panoptic_150)
plt.close()

pre_unique = np.unique(panoptic_150)
print(pre_unique)
#for i in pre_unique:




percent_80 = 0
percent_20 = 0

new_object = cv2.imread(workspace + 'mask_0017.png', 0)

for i in pre_unique:
    mask = i == panoptic_150
    #print(mask)
    print("\n\n\nid : ",i)
    
    mask_1D = mask.reshape(-1)
    #mask_1D = mask.flatten()
    #print(mask_1D)
    #print(mask_1D.shape)


    count_all = np.count_nonzero(mask_1D == True)
    print(count_all)
    

    dishow(mask)
    plt.close()
    

    if (count_all < 1000):
        mask = 1 - mask
        panoptic_150 = panoptic_150 * mask
        continue
    
    if (i > 0):

        #mask는 boolean, mask_occlusion은 0과 255

        mask_occlusion = mask * new_object
        
        dishow(mask_occlusion)
        plt.close()

        mask_occlusion_1D = mask_occlusion.reshape(-1)
        count_occlusion = np.count_nonzero(mask_occlusion_1D == 255)
        print(count_occlusion)

        percent = (count_occlusion / count_all)
        print(percent)

        if (percent >= 80):
            percent_80 = percent_80 + 1       

        if (percent <= 20):
            percent_20 = percent_20 + 1







    


dishow(panoptic_150)
plt.close()


pre_unique = np.unique(panoptic_150)
print(pre_unique)
print(len(pre_unique))

print("ing")


print(percent_80)
print(percent_20)


if (percent_80 >= 1):
    print("x")
if (percent_20 == 0):
    print("x")



'''
for dir in folder_list:
    for i in range(20):
        workspace = folder_root + folder_list[0] + '\\'
        print(workspace)
        print(i)

        rgb_name = workspace + 'rgb_' + str(i).zfill(4) + '.png'
        depth_name = workspace + 'depth_' + str(i).zfill(4) + '.exr'
        
        segmap_name = workspace + 'segmap_' + str(i).zfill(4) + '.npy'

        
        depth = pyexr.read(depth_name)
        depth = (depth-np.min(depth))/(np.max(depth)-np.min(depth))
        depth = depth * 255
        depth = np.asarray(depth, dtype=int)

        cv2.imwrite(workspace + 'LDY_depth_' + str(i).zfill(4) + '.png', depth)
        


        # segmentic 0 : 가구 다똑같음
        # instance segmentaion 1 : mask 다 넘버링 다름


        segmap = np.load(segmap_name)

        segmap0 = segmap[:,:,0]
        pd.DataFrame(segmap0).to_csv(workspace + 'LDY_Excel_segmap0_' + str(i).zfill(4) + '.csv')
        segmap0 = (segmap0-np.min(segmap0))/(np.max(segmap0)-np.min(segmap0))
        segmap0 = segmap0 * 255
        segmap0 = np.asarray(segmap0, dtype=int)
        cv2.imwrite(workspace + 'LDY_segmap0_' + str(i).zfill(4) + '.png', segmap0)
        



        segmap1 = segmap[:,:,1]
        pd.DataFrame(segmap1).to_csv(workspace + 'LDY_Excel_segmap1_' + str(i).zfill(4) + '.csv')
        cv2.imwrite(workspace + 'LDY_segmap1_' + str(i).zfill(4) + '.png', segmap1)
'''
# %%
