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

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


import psutil
import gc


def dishow(disp):
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    plt.show()
    plt.savefig(fname=workspace +'image2.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()


def dishow_LDY(disp, i):
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    #plt.show()

    plt.savefig(fname=workspace +'panoptic_' + str(i).zfill(4) + '.jpg', bbox_inches='tight', pad_inches=0)
    #plt.savefig(fname= root+dir + '\\' +'panoptic_' + str(i).zfill(4) + '.jpg', bbox_inches='tight', pad_inches=0)


    plt.close()





# load MaskFormer fine-tuned on COCO panoptic segmentation




root= 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\data230306_filtering\\'
file_list = os.listdir(root)

file_list = [file for file in file_list if not file.endswith(".txt")]

print(file_list)
print(len(file_list))

image_processor = AutoImageProcessor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")


m = 0
for dir in file_list:

    workspace = root + dir + '\\'

    print(m, " : ", len(file_list))
    m = m + 1
    print(dir)

    #작업 완료 폴더 건너뛰기
    checking = workspace + 'panoptic_0018.jpg'
    if (os.path.isfile(checking)):
        continue

    for i in range(20):

        if i%2 > 0:
            continue

        #url = "http://images.cocodataset.org/val2017/000000039769.jpg"


        

        #print(workspace)
        #print(i)
        url = workspace + 'rgb_' + str(i).zfill(4)  + '.png'

        #image = Image.open(requests.get(url, stream=True).raw)
        image = Image.open(url)


        
        inputs = image_processor(images=image, return_tensors="pt")

        outputs = model(**inputs)
        

       

      

        

        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`


        #class_queries_logits = outputs.class_queries_logits
        #masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to image_processor for postprocessing
        result = image_processor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

        # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        predicted_panoptic_map = result["segmentation"]
        list(predicted_panoptic_map.shape)

        #print(list(predicted_panoptic_map.shape))
        #print(type(predicted_panoptic_map))

       
        # -------------------- 메모리 관리 !!! -------------------------
        del outputs

        output = predicted_panoptic_map.numpy()
        cv2.imwrite(workspace +'panoptic_origin_' + str(i).zfill(4)  + '.png', output)

        dishow_LDY(output, i)


        output_norm = (output-np.min(output))/(np.max(output)-np.min(output))
        output_norm = output_norm * 255
        output_norm = np.asarray(output_norm, dtype=int)
        cv2.imwrite(workspace +'panoptic_norm_' + str(i).zfill(4)  + '.png', output_norm)
        


        '''
        del inputs
        del image_processor
        del outputs
        del model
        #gc.collect()

        
        # AFTER  code
        memory_usage_dict = dict(psutil.virtual_memory()._asdict())
        memory_usage_percent = memory_usage_dict['percent']
        print(f"AFTER  CODE: memory_usage_percent: {memory_usage_percent}%")
        # current process RAM usage
        pid = os.getpid()
        current_process = psutil.Process(pid)
        current_process_memory_usage_as_KB = current_process.memory_info()[0] / 2.**20
        print(f"AFTER  CODE: Current memory KB   : {current_process_memory_usage_as_KB: 9.3f} KB")

        print("--"*30)
        '''



#print(np.min(output))
#print(np.max(output))


#dishow(output)

#cv2.imwrite('sample.png', output)
#[480, 640]

# %%
