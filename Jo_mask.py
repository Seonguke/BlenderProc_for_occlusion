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
    #plt.savefig(fname='image.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()
def get_mask(pre_mask):
    pre_mask = []
    pre_unique = np.unique(pre[..., 0])
    print(pre_unique)
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

        for j in unique:
            small_mask = markers == j
            if small_mask.sum() < 200 or small_mask.sum() >= max_area:
                continue
            pre_mask.append(small_mask)
    return pre_mask
pre = np.load('C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230307\\(img1)segmap_0016.npy')
nxt = np.load('C:\\Users\\lab-com\\Desktop\\myspace\\BlenderProc_for_occlusion\\test230307\\(img1)segmap_0017.npy')
dishow(pre[:,:,0])
dishow(nxt[:,:,0])
pre_mask = get_mask(pre)
#nxt_mask = get_mask(nxt)
for mask in pre_mask:
    dishow(mask)
'''
for mask in nxt_mask:
    dishow(mask)
'''

# unique_ids = np.unique(semantic_label)
# unique_ids = np.unique(semantic_label2)