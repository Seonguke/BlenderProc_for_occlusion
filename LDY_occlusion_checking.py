import os
import random

import cv2
import pyexr

import numpy as np

import pandas as pd 


from typing import Dict, Optional, List, Tuple




folder_root = 'C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230302\\'

#folder_list = os.listdir(folder_root)
folder_list = ['002c110c-9bbc-4ab4-affa-4225fb127bad']


#print(folder_list)
print("folder files : ", len(folder_list))


'''
class SegmentationToMasks:
    def __init__(self, image_size: Tuple[int, int], num_min_pixels: int = 200, max_instances: int = None,
                 shuffle_instance_ids: bool = False, ignore_classes: List[int] = None):
        self.image_size = image_size
        self.num_min_pixels = num_min_pixels
        self.max_instances = max_instances
        self.shuffle_instance_ids = shuffle_instance_ids
        #self.colormap=


        if ignore_classes is None:            ignore_classes = []
        self.ignore_classes = ignore_classes

    def __call__(self, segmentation_image: np.array):
        # Segmentation file stores at channels
        # 0: semantic segmentation
        # 1: instance segmentation
        semantic_image = segmentation_image[..., 0]
        instance_image = segmentation_image[..., 1]

        labels = []
        bounding_boxes = []
        masks = []
        unique_ids = np.unique(instance_image).astype(np.uint32)

        # Iterate over all unique instances
        enumerated_instance_indices = {}
        instance_mapping = {}

        # Start at index 1 to leave space for 3D freespace 0-label
        randomized_indices = list(range(1, self.max_instances + 1))

        if self.shuffle_instance_ids:
            random.shuffle(randomized_indices)

        instance_ids = []

        # Manually count in case instances are skipped
        instance_counter = 0

        #LDY
        #unique_ids = [65]

        for instance_id in unique_ids:
            # Stop when max valid instances are found
            if instance_counter >= len(randomized_indices):
                break

            # Get current instance mask
            instance_mask: np.array = instance_image == instance_id
            num_instance_pixels = np.sum(instance_mask)
            instance_coordinates = instance_mask.nonzero()

            if num_instance_pixels <= self.num_min_pixels:
                continue

            # Determine semantic label of the current instance
            semantic_labels = semantic_image[instance_coordinates[0], instance_coordinates[1]]
            unique_semantic_labels, semantic_label_count = np.unique(semantic_labels, return_counts=True)
            max_semantic_label = np.argmax(semantic_label_count)
            semantic_label = unique_semantic_labels[max_semantic_label]

            if semantic_label in self.ignore_classes:
                continue

            # Compute bounding box
            min_y, min_x = np.min(instance_coordinates[0]), np.min(instance_coordinates[1])
            max_y, max_x = np.max(instance_coordinates[0]), np.max(instance_coordinates[1])
            bbox2d = [min_x, min_y, max_x, max_y]

            labels.append(semantic_label)
            bounding_boxes.append(bbox2d)
            masks.append(instance_mask)
            enumerated_instance_indices[int(instance_id)] = instance_counter

            random_index = randomized_indices[instance_counter]
            instance_ids.append(random_index)
            instance_mapping[int(instance_id)] = random_index

            instance_counter += 1

        print("end")
        
        
        indices = instance_mask.astype(np.uint8)  #convert to an unsigned byte
        indices*=255
        #cv2.imshow('Indices',indices)
        cv2.imwrite(folder_root + test_folder + "bbb.png", indices)
        cv2.imwrite(folder_root + test_folder + "ccc.png", semantic_image)
        
        



test_folder = 't230301\\'
a = folder_root + test_folder + "segmap_0006.npy"
#bounding_boxes.add_field("mask2d", SegmentationMask(torch.from_numpy(np.array(masks)), self.image_size, mode="mask"))
#b = SegmentationMask(torch.from_numpy(np.array(masks)), self.image_size, mode="mask")

b = SegmentationToMasks(image_size = [240, 320], max_instances=200)
print(b)

a = np.load(a)

a1_0 = a[:,:,0].copy()
a1_1 = a[:,:,1].copy()


print(a.shape)
print(a[:,:,0].shape)
c = b(a)
print(b)
print(c)

#print(b.shape)

print(a.shape)
print(a[:,:,0].shape)
print(a[:,:,0])

print(a[:,:,1].shape)
print(a[:,:,1])

print("-----------------------")

a2_0 = a[:,:,0].copy()
a2_1 = a[:,:,1].copy()





print(np.array_equal(a1_0,a1_0))
print(np.array_equal(a1_0,a2_0))
print(np.array_equal(a1_1,a2_1))

#print(c.shape)

cv2.imwrite(folder_root + test_folder + "aaa.png", a[:,:,1])
pd.DataFrame(a[:,:,1]).to_csv(folder_root + test_folder +  'aaa.csv')

#--------------------------------------------------------------------------------
'''
print("start")


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
        segmap0 = segmap[:,:,0]
        test = segmap[:,:,1]
        for m in range(240):
            for n in range(320):
                if ((segmap0[m][n] == 0) or (segmap0[m][n] == 1)):
                    test[m][n] == 0
        pd.DataFrame(test).to_csv(workspace + 'LDY_test_' + str(i).zfill(4) + '.csv')
        cv2.imwrite(workspace + 'LDY_test_' + str(i).zfill(4) + '.png', test)
        '''