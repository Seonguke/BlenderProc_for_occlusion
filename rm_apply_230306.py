import os
import cv2
import pyexr
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

import json



root= 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\data230306_filtering\\'
file_list = os.listdir(root)

file_list = [file for file in file_list if not file.endswith(".txt")]

print(file_list)

#file_list = ['t1']

num = 0
for dir in file_list:

    print(dir)
    print(num ," : ", len(file_list))
    num = num + 1


    workspace = root + dir + '\\'
    file_path = workspace + dir + '_rmlist.json'

    using_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)


    for k in range(0, 20):
        #print(k)
        
        if k%2 > 0:
            continue

        if (data[k][0] == 'False'):
            #print("False")
            continue
        

        #k번
        rgb = 'rgb_' + str(k).zfill(4) + '.png'
        depth = 'depth_' + str(k).zfill(4) + '.exr'
        using_list.append(rgb)
        using_list.append(depth)


        depth = workspace + depth
        #depth값 이상한거 변경
        if(os.path.isfile(depth)):
            #print("file exist")
            depth_value = pyexr.read(depth)
            max_value = np.max(depth_value)
            #print(max_value)
            if(max_value > 10.0):
                print("outlier")
                depth_value_processing = np.where(depth_value > 10, 10, depth_value)
                pyexr.write(depth, depth_value_processing)


        #k+1번
        rgb = 'rgb_' + str(k+1).zfill(4) + '.png'
        depth = 'depth_' + str(k+1).zfill(4) + '.exr'
        using_list.append(rgb)
        using_list.append(depth)

        depth = workspace + depth
        #depth값 이상한거 변경
        if(os.path.isfile(depth)):
            #print("file exist")
            depth_value = pyexr.read(depth)
            max_value = np.max(depth_value)
            #print(max_value)
            if(max_value > 10.0):
                #print("outlier")
                depth_value_processing = np.where(depth_value > 10, 10, depth_value)
                pyexr.write(depth, depth_value_processing)






    using_list.append(os.path.basename(file_path))





    '''
    print("------------------------------")

    for e in using_list:
        print(e)

    print("------------------------------")
    '''
    all_list = os.listdir(workspace)

    
    '''
    #파일 삭제
    #rm_list = os.path.isfile(workspace)
    for e in all_list:
        print(e)
    '''



    rm_list = list(set(all_list).difference(using_list))
    #rm_list = [x for x in all_list if x not in using_list]

    rm_list.sort()
    '''
    for e in rm_list:
        print(e)

    print("------------------------------")
    '''
    for e in rm_list:
        #print(e)
        e =  workspace + '\\' + e
        #print(e)
        if (os.path.isfile(e)):
            os.remove(e)
        elif(os.path.isdir(e)):
            shutil.rmtree(e)


    '''
    for e in rm_list:
        print(e)
        e =  workspace + '\\' + e
        print(e)
        os.remove(e)


    #폴더 삭제
    rm_list = os.path.isdir(workspace)
    for e in rm_list:
        print(e)
        e =  workspace + '\\' + e
        print(e)
        shutil.rmtree(e)
    '''





        
