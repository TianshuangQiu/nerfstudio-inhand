import json
import os

import cv2
import numpy as np

dict_path = "/home/ethantqiu/data/ice_cream_dec11_second/transforms.json" 

with open(dict_path, 'r') as f:
    data = json.load(f)

frames = []
for f in data["frames"]:
    f["file_path"] = f["file_path"].replace("data/ice_cream_dec11_second/", "")
    
    masked_im = f["mask_path"]
    mask = cv2.imread(masked_im)
    mask = np.sum(mask, axis=2)
    mask = mask > 0
    mask = mask.astype(np.uint8)
    cv2.imwrite(masked_im, mask*255)
    frames.append(f)
    
    f["mask_path"] = f["mask_path"].replace("data/ice_cream_dec11_second/", "")
data["frames"] = frames

with open(dict_path, 'w') as f:
    json.dump(data, f, indent=4)