#%%
from typing import Dict, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

from transformers import AutoImageProcessor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests

import os

def dishow(disp):
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    plt.show()
    plt.savefig(fname=workspace +'image2.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()


def dishow_LDY(disp, i):
    #plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    #plt.show()

    #plt.savefig(fname=workspace +'panoptic_' + str(i).zfill(4) + '.jpg', bbox_inches='tight', pad_inches=0)
    plt.savefig(fname= root+dir + '\\' +'panoptic_' + str(i).zfill(4) + '.jpg', bbox_inches='tight', pad_inches=0)


    plt.close()








workspace = './test230302/002c110c-9bbc-4ab4-affa-4225fb127bad/'


image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

root= 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230303_sample\\'
file_list = os.listdir(root)
for dir in file_list:
    for i in range(20):
        if i%2 > 0:
            continue
        else:
            print(dir)
            name = root+dir + '\\' + 'rgb_' + str(i).zfill(4) + '.png'
            print(i, " : 19")
            
            #image = Image.open(requests.get(url, stream=True).raw)
            image = Image.open(name)

            inputs = image_processor(images=image, return_tensors="pt")

            outputs = model(**inputs)
            # model predicts class_queries_logits of shape `(batch_size, num_queries)`
            # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
            class_queries_logits = outputs.class_queries_logits
            masks_queries_logits = outputs.masks_queries_logits

            # you can pass them to image_processor for postprocessing
            result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

            # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
            predicted_panoptic_map = result["segmentation"]
            list(predicted_panoptic_map.shape)

            #print(list(predicted_panoptic_map.shape))
            #print(type(predicted_panoptic_map))

            output = predicted_panoptic_map.numpy()
        
            
            cv2.imwrite(root+dir + '\\' + 'panoptic_origin_' + str(i).zfill(4)  + '.png', output)

            dishow_LDY(output, i)


            output_norm = (output-np.min(output))/(np.max(output)-np.min(output))
            output_norm = output_norm * 255
            output_norm = np.asarray(output_norm, dtype=int)
            #cv2.imwrite(workspace +'panoptic_norm_' + str(i).zfill(4)  + '.png', output_norm)


'''
# load MaskFormer fine-tuned on COCO panoptic segmentation
image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

for i in range(20):
    #url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(i)
    url = workspace + 'rgb_' + str(i).zfill(4)  + '.png'

    #image = Image.open(requests.get(url, stream=True).raw)
    image = Image.open(url)

    inputs = image_processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    # model predicts class_queries_logits of shape `(batch_size, num_queries)`
    # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
    class_queries_logits = outputs.class_queries_logits
    masks_queries_logits = outputs.masks_queries_logits

    # you can pass them to image_processor for postprocessing
    result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    predicted_panoptic_map = result["segmentation"]
    list(predicted_panoptic_map.shape)

    #print(list(predicted_panoptic_map.shape))
    #print(type(predicted_panoptic_map))






    output = predicted_panoptic_map.numpy()
    cv2.imwrite(workspace +'panoptic_origin_' + str(i).zfill(4)  + '.png', output)

    dishow_LDY(output, i)


    output_norm = (output-np.min(output))/(np.max(output)-np.min(output))
    output_norm = output_norm * 255
    output_norm = np.asarray(output_norm, dtype=int)
    #cv2.imwrite(workspace +'panoptic_norm_' + str(i).zfill(4)  + '.png', output_norm)


#print(np.min(output))
#print(np.max(output))


#dishow(output)

#cv2.imwrite('sample.png', output)
#[480, 640]
'''
# %%
