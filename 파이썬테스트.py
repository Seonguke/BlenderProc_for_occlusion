
import json
import os

import numpy as np

import cv2

'''
#json file 생성
json_list = []

for i in range(0, 30):
    if((i%3)==0):
        index = '' + str(i) + ''
        json_list.append(index)
print("\n\n\n\n\n\n", json_list)

output_dir = './output_JSON_LDY'
json_name = os.path.basename(output_dir)
json_path = "C:\\Users\\lab-com\\Desktop\\myspace\\\BlenderProc_for_occlusion\\output_JSON_LDY\\" + str(json_name) + ".json"

with open(json_path, 'w') as f:
    json.dump(json_list, f, indent="\t")
'''

'''
a = "./output_LDY/0003d406-5f27-4bbf-94cd-1cff7c310ba1/depth_0000.exr"

print(a)
'''


'''
b = np.load("./output_LDY/00154c06-2ee2-408a-9664-b8fd74742897/segmap_0002.npy")

print(b.shape)

print("1")
# instance segmentaion 1 - mask 다 넘버링 다름
# segmentic 0 - 가구 다똑같음
'''

print("EXR")

