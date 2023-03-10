#%%
from typing import Dict, Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2

def dishow(disp):
    plt.imshow(disp)
    plt.jet()
    plt.colorbar()
    plt.plot
    plt.show()
    #plt.savefig(fname= workspace +'image.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

def get_mask(pre_mask):
    pre_mask = []
    pre_unique = np.unique(pre[..., 1])
    for i in pre_unique:

        mask = i == pre[..., 1]
        #if i==1 or i==2:
        #    pre_mask.append(mask.astype(bool))
        #    continue
        h, w = mask.shape
        max_area = (h * w) - mask.sum()
        mask = mask.astype(np.uint8) * 255
        ret, markers = cv2.connectedComponents(mask)
        markers = markers + 1
        unique = np.unique(markers)

        for j in unique:
            small_mask = markers == j
            if small_mask.sum() < 200 or small_mask.sum() >= max_area:
                continue
            pre_mask.append(small_mask)
            #print("1")
    return pre_mask

workspace = './test230302/002c110c-9bbc-4ab4-affa-4225fb127bad/'

pre = np.load(workspace + 'segmap_0016.npy')
nxt = np.load(workspace + 'segmap_0017.npy')
dishow(pre[:,:,1])
dishow(nxt[:,:,0])
re_unique = np.unique(pre[..., 1])
#for i in re_unique:
#    dishow(pre[:,:,1]==i)
pre_mask = get_mask(pre)
nxt_mask = get_mask(nxt)
for mask in pre_mask:
    dishow(mask)
#for mask in nxt_mask:
#    dishow(mask)

# unique_ids = np.unique(semantic_label)
# unique_ids = np.unique(semantic_label2)
# %%
