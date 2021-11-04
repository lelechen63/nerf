import random
import numpy as np
import imageio
import cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os

def get_rays_np_cvcam(H, W, K, Rt, scale=1):
    """
    (H, W): size of the imaging film
    K: intrinsic matrix (3x3)
    Rt: extrinsic matrix (3x4), world to camera
    """
    K = np.array(K)
    Rt = np.array(Rt)
    H = H * scale
    W = W * scale
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0] * scale, K[1][1] * scale
    cx, cy = K[0][2] * scale, K[1][2] * scale
    
    c2w = np.eye(4)
    c2w[:3, :3] = Rt[:3, :3].T
    c2w[:3, 3] = -Rt[:3, :3].T.dot(Rt[:, 3])

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cx)/fx, -(j-cy)/fy, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

class FacescapeLoader(Dataset):
    def __init__(self, dataRoot, face_id, exp_id, scale=1.0):
        rays_path = f"{face_id}_{exp_id}_s{scale}_rays.npy"
        if os.path.exists(rays_path):
            self.rays_base = np.load(rays_path)
        else:
            faceRoot = f'{dataRoot}/fsmview_images/{face_id}/{exp_id}'
            with open(f'{faceRoot}/params.json') as f:
                params = json.load(f)        
            img_id = 0
            self.rays_base = []
            while f'{img_id}_K' in params:
                print(f"processing {img_id}")
                valid = params[f'{img_id}_valid']
                if valid:
                    K = np.array(params[f'{img_id}_K'])
                    Rt = np.array(params[f'{img_id}_Rt'])
                    W = params[f'{img_id}_width']
                    H = params[f'{img_id}_height']
                    dist = np.array(params[f'{img_id}_distortion'])
                    img_path = f'{faceRoot}/{img_id}.jpg'
                    img = (imageio.imread(img_path) / 255.0).astype(np.float32) # [H, W, 3]
                    rays = np.stack(get_rays_np_cvcam(H, W, K, Rt, scale), 0) # [ro+rd, H, W, 3]
                    img_undist = cv2.undistort(img, K, dist) # [H, W, 3]
                    img_undist = cv2.resize(img_undist, (rays.shape[2], rays.shape[1]))
                    rays = np.concatenate([rays, img_undist[np.newaxis,...]], 0) # [ro+rd+rgb, H, W, 3]
                    rays = rays.transpose([1,2,0,3]) # [H, W, ro+rd+rgb, 3]
                    rays = rays.reshape([-1,3,3]) # [H*W, ro+rd+rgb, 3]
                    self.rays_base.append(rays)
                img_id = img_id + 1
            
            self.rays_base = np.concatenate(self.rays_base, 0)
            self.rays_base = self.rays_base.astype(np.float32)
            np.save(rays_path, self.rays_base)
            print(f"precomputed rays saved in {rays_path}")
        
        print(f'{self.rays_base.shape[0]} pixel loaded, type {self.rays_base.dtype}')
        np.random.shuffle(self.rays_base)
    
    def __len__(self):
        return self.rays_base.shape[0]
    
    def __getitem__(self, ind):
        batch_dict = {
            "rgb_rays": self.rays_base[ind]
        }
        return batch_dict


if __name__ == '__main__':
    loader = FacescapeLoader("/home/goddice/Work/nerf-space/data/facescape", 183, "17_cheek_blowing", scale=0.1)