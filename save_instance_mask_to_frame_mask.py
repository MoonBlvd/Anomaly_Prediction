import os
import numpy as np
import cv2
import torch
import glob
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
import time
from tqdm import tqdm
import pdb
root = '/home/data/vision7/A3D_2.0' 

def det_to_mask(det):
    # remove traffic lights/signs/trains
    keep = []
    for i in range(len(det)):
        if det.extra_fields['labels'][i] <8:
            keep.append(i)
    det.bbox = det.bbox[keep]
    for k in det.extra_fields.keys():
        det.extra_fields[k] = det.extra_fields[k][keep]

    masks = det.get_field("mask")
    # end = time.time()
    masks = masker([masks], [det])[0]
    # print("time:", time.time()-end)
    masks = masks.sum(dim=0).clamp(min=0, max=1).permute(1,2,0).numpy()
    masks = masks.astype(np.uint8)
    masks = cv2.resize(masks, (256, 256)).astype('uint8')
    
    return masks

if __name__ == '__main__':
    masker = Masker(threshold=0.5, padding=1)

    all_folders = sorted(glob.glob(os.path.join(root, 'detection_with_seg', '*')))
    for folder in tqdm(all_folders):
        vid = folder.split('/')[-1]
        save_path = os.path.join(root, 'foreground_segments', vid+'.npy')

        if os.path.exists(save_path):
            print("{} has been processed!".format(vid))
            continue
        all_det_files = sorted(glob.glob(os.path.join(folder, '*.pth')))
        per_video_masks = []
        for det_file in all_det_files:
            det = torch.load(det_file)
            mask = det_to_mask(det)
            per_video_masks.append(mask)
        try:
            per_video_masks = np.stack(per_video_masks, axis=0)
        except:
            pdb.set_trace()
        
        np.save(save_path, per_video_masks)
