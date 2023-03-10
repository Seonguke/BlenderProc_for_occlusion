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
    #print("show")
    
    
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    plt.show()
    #plt.savefig(fname=workspace + 'show/' + 'test.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()
    


def mask_show(panoptic_150, pre_unique):

    print("---------------- showing objects ----------------")


    dishow(panoptic_150)

    for i in pre_unique:
            mask = i == panoptic_150
            #print(mask)
            print("\n\n\nid : ",i)
            
            mask_1D = mask.reshape(-1)
            count_all = np.count_nonzero(mask_1D == True)
            print("point 개수 : " ,count_all)


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

    

#i = 18



root= 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230302\\'
file_list = os.listdir(root)

file_list = [file for file in file_list if not file.endswith(".txt")]

print(file_list)

#file_list = ['00ecd5d3-d369-459f-8300-38fc159823dc']


'''
workspace ='C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230302\\002c110c-9bbc-4ab4-affa-4225fb127bad\\'

segmap = np.load(workspace + 'segmap_0016.npy')

segmap0 = segmap[:,:,0]
segmap1 = segmap[:,:,1]

dishow(segmap0)
dishow(segmap1)
'''


for dir in file_list:


    #json file 생성
    json_list = []
    print(dir)

    workspace = root + dir + '\\'

    workspace ='C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230307\\(img1)'


    for k in range(0, 20):

        k = 16

        print("------------- 1 -----------------------")
        print(workspace)
        print(k)


        index = '' + str(k) + ''
        index_inform = []
        index_inform.append(index)

        if k%2 > 0:
            #index_inform.append("Pass")
            index_inform.insert(0, "Pass")
            json_list.append(index_inform)
            continue


        rm_count = 0


        pixel_count = 320 * 240

        print(workspace + 'mask_' + str(k+1).zfill(4)  + '.png', 0)
        new_object_mask = cv2.imread(workspace + 'mask_' + str(k+1).zfill(4)  + '.png', 0)
        dishow(new_object_mask)


        new_object_mask_1D = new_object_mask.reshape(-1)
        new_object_mask_count = np.count_nonzero(new_object_mask_1D == 255)
        print(new_object_mask_count)

        percent_new_object_mask = (new_object_mask_count / pixel_count)
        print(percent_new_object_mask)

        if (percent_new_object_mask > 0.75):
            print("a : ", percent_new_object_mask)
            print("x")
            rm_count = rm_count + 1
            rm_inform = "마스크 너무 큼 Too big percent_new_object_mask : " + str(percent_new_object_mask)
            index_inform.append(rm_inform)




        if (percent_new_object_mask < 0.03):
            print("마스크 너무 작음 Too small percent_new_object_mask : ", percent_new_object_mask)
            print("x")
            rm_count = rm_count + 1
            rm_inform = "마스크 너무 작음 Too small percent_new_object_mask : " + str(percent_new_object_mask)
            index_inform.append(rm_inform)


        segmap = np.load(workspace + 'segmap_' + str(k).zfill(4) + '.npy')
        segmap0 = segmap[:,:,0]
        #plt.imshow(segmap0)
        dishow(segmap0)



        segmap0_unique = np.unique(segmap0)
        print(segmap0_unique)
        print(len(segmap0_unique))

        if (len(segmap0_unique) < 3):
            print('x')
            rm_count = rm_count + 1
            rm_inform = "no object in first"
            index_inform.append(rm_inform)

        segmap1 = segmap[:,:,1]
        #plt.imshow(segmap0)
        dishow(segmap1)


        segmap1_unique = np.unique(segmap1)
        print(segmap1_unique)
        print(len(segmap1_unique))

        mask_show(segmap1, segmap1_unique)

        #new_object_pre_unique = np.unique(new_object_mask)
        #print(new_object_pre_unique)


        #panoptic_origin = cv2.imread(workspace + 'panoptic_origin_0016.png')
        panoptic_origin = cv2.imread(workspace + 'panoptic_origin_' + str(k).zfill(4) + '.png', 0)

        #segmap1로 테스트
        #panoptic_origin = segmap[:,:,1]


        #plt.imshow(panoptic_origin)
        dishow(panoptic_origin)



        # -------------------- panoptic 결과에 + 10  ----------------------- #


        #panoptic_150 = panoptic_origin + 150
        panoptic_150 = panoptic_origin + 10
        #panoptic_150 = panoptic_origin
        #plt.imshow(panoptic_150)
        dishow(panoptic_150)

        

        pre_unique = np.unique(panoptic_150)
        print(pre_unique)
        mask_show(panoptic_150, pre_unique)



        panoptic_150 = panoptic_150 + segmap1
        dishow(panoptic_150)
        pre_unique = np.unique(panoptic_150)
        print(pre_unique)
        mask_show(panoptic_150, pre_unique)





        '''
        panoptic_150 = count_filter(panoptic_150, pre_unique)
        pre_unique = np.unique(panoptic_150)
        print(pre_unique)
        dishow(panoptic_150)
        mask_show(panoptic_150, pre_unique)
        '''


        # -------------------- 벽바닥 마스킹 과정 시작 ----------------------- #

        # 1이 벽, 2가 바닥
        segmap_mask_2 = np.where(segmap0 == 2, 0, 1)
        #plt.imshow(segmap_mask_2)
        dishow(segmap_mask_2)



        panoptic_150 = panoptic_150 * segmap_mask_2
        #plt.imshow(panoptic_150)
        dishow(panoptic_150)



        panoptic_150 = np.where(panoptic_150 == 0, 2, panoptic_150)
        dishow(panoptic_150)



        segmap_mask_1 = np.where(segmap0 == 1, 0, 1)
        #plt.imshow(segmap_mask_1)
        dishow(segmap_mask_1)



        panoptic_150 = panoptic_150 * segmap_mask_1
        #plt.imshow(panoptic_150)
        dishow(panoptic_150)


        panoptic_150 = np.where(panoptic_150 == 0, 1, panoptic_150)
        dishow(panoptic_150)




        pre_unique = np.unique(panoptic_150)
        print(pre_unique)
        mask_show(panoptic_150, pre_unique)


        # -------------------- 벽바닥 마스킹 과정 끝 ----------------------- #



        print('----------------------------------------')

        percent_60 = 0
        percent_15 = 0



        for i in pre_unique:
            mask = i == panoptic_150
            #print(mask)
            print("\n\n\nid : ",i)
            
            mask_1D = mask.reshape(-1)
            #mask_1D = mask.flatten()
            #print(mask_1D)
            #print(mask_1D.shape)



            
            if (i == 2):

                dishow(mask)


                mask_occlusion = mask * new_object_mask
                mask_occlusion_1D = mask_occlusion.reshape(-1)
                count_occlusion = np.count_nonzero(mask_occlusion_1D == 255)
                print(count_occlusion)

                count_all = np.count_nonzero(mask_1D == True)
                print(count_all)

                dishow(mask_occlusion)


                percent = (count_occlusion / count_all)
                print(percent)
                if (percent >= 0.8):
                    print("바닥 너무 많이 가림 : ", percent)
                    print("x")
                    rm_count = rm_count + 1
                    rm_inform = "바닥 너무 많이 가림 : " + str(percent)
                    index_inform.append(rm_inform)

                    
            if (i > 2):

                count_all = np.count_nonzero(mask_1D == True)
                print(count_all)
                

                dishow(mask)



                if (count_all < 1000):
                    mask = 1 - mask
                    panoptic_150 = panoptic_150 * mask
                    continue


                #mask는 boolean, mask_occlusion은 0과 255

                mask_occlusion = mask * new_object_mask
                
                dishow(mask_occlusion)


                mask_occlusion_1D = mask_occlusion.reshape(-1)
                count_occlusion = np.count_nonzero(mask_occlusion_1D == 255)
                print(count_occlusion)

                percent = (count_occlusion / count_all)
                print(i , " percent : ", percent)

                if (percent >= 0.6):
                    print("60퍼이상 겹침")
                    percent_60 = percent_60 + 1       

                if (percent >= 0.15):
                    print("15퍼 이상 겹침")
                    percent_15 = percent_15 + 1



        dishow(panoptic_150)



        pre_unique = np.unique(panoptic_150)
        print(pre_unique)
        print(len(pre_unique))

        if (len(pre_unique) < 2):
            print("물체가 없음")
            print("x")
            rm_count = rm_count + 1
            rm_inform = "물체가 없음 : " + str(pre_unique)
            index_inform.append(rm_inform)

        print("ing")


        print(percent_60)
        print(percent_15)


        print("---------------------")

        if (percent_60 >= 1):
            print("60퍼 이상 가림 : ", percent_60)
            print("x")
            rm_count = rm_count + 1
            rm_inform = "60퍼 이상 가림 :" + str(percent_60)
            index_inform.append(rm_inform)
        if (percent_15 == 0):
            print("15퍼 이상 가림 : ", percent_15)
            print("x")
            rm_count = rm_count + 1
            rm_inform = "15퍼 이상 가림 : " + str(percent_15)
            index_inform.append(rm_inform)



        if (rm_count > 0):
            print('지워야 함 : ', rm_count)
            print("remove")
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

        print(json_list)
    
    print(json_list)



    #json_name = os.path.basename(dir)
    print(workspace)
    json_path = workspace + dir  +"_rmlist" + ".json"
    '''
    with open(json_path, 'w') as f:
        json.dump(json_list, f, indent="\t")
    '''
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_list, f, indent="\t", ensure_ascii=False)
