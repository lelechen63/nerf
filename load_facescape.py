import numpy as np
import imageio
import cv2
import json
from tqdm import tqdm

def get_rays_np_cvcam(H, W, K, Rt, scale=1, ndc=True):
    """
    (H, W): size of the imaging film
    K: intrinsic matrix (3x3)
    Rt: extrinsic matrix (3x4), world to camera
    """
    K = np.array(K)
    Rt = np.array(Rt)
    H = int(H * scale)
    W = int(W * scale)
    # Caculate fx fy cx cy from K
    fx, fy = K[0][0] * scale, K[1][1] * scale
    cx, cy = K[0][2] * scale, K[1][2] * scale
    
    c2w = np.eye(4)
    c2w[:3, :3] = Rt[:3, :3].T
    c2w[:3, 3] = -Rt[:3, :3].T.dot(Rt[:, 3])

    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-cx)/fx, (j-cy)/fy, np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))

    if ndc:
        near = 1.0
        # Shift ray origins to near plane
        # t = -(near + rays_o[...,2]) / rays_d[...,2]
        # rays_o = rays_o + t[...,None] * rays_d
        
        # # Projection
        # o0 = -fx/cx * rays_o[...,0] / rays_o[...,2]
        # o1 = -fy/cy * rays_o[...,1] / rays_o[...,2]
        # o2 = 1. + 2. * near / rays_o[...,2]

        # d0 = -fx/cx * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
        # d1 = -fy/cy * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
        # d2 = -2. * near / rays_o[...,2]
        
        # rays_o = np.stack([o0,o1,o2], -1)
        # rays_d = np.stack([d0,d1,d2], -1)
    return rays_o, rays_d

def load_facescape_data(basedir, scale=1.0, ndc=True):
    K_list = []
    Rt_list = []
    W_list = []
    H_list = []

    with open(f'{basedir}/params.json') as f:
        params = json.load(f)        
    img_id = 0
    rays_base = []
    while f'{img_id}_K' in params:
        print(f"processing {img_id}")
        valid = params[f'{img_id}_valid']
        if valid:
            K = np.array(params[f'{img_id}_K'])
            Rt = np.array(params[f'{img_id}_Rt'])
            W = params[f'{img_id}_width']
            H = params[f'{img_id}_height']
            dist = np.array(params[f'{img_id}_distortion'])
            img_path = f'{basedir}/{img_id}.jpg'
            img = (imageio.imread(img_path) / 255.0).astype(np.float32)
            rays = np.stack(get_rays_np_cvcam(H, W, K, Rt, scale, ndc), 0) # [ro+rd, H, W, 3]
            
            img_undist = cv2.undistort(img, K, dist) # [H, W, 3]
            img_undist = cv2.resize(img_undist, (rays.shape[2], rays.shape[1]))

            K_list.append(K)
            Rt_list.append(Rt)
            W_list.append(rays.shape[2])
            H_list.append(rays.shape[1])

            rays = np.concatenate([rays, img_undist[np.newaxis,...]], 0) # [ro+rd+rgb, H, W, 3]
            rays = rays.transpose([1,2,0,3]) # [H, W, ro+rd+rgb, 3]
            rays = rays.reshape([-1,3,3]) # [H*W, ro+rd+rgb, 3]
            rays_base.append(rays)
        img_id = img_id + 1
    
    rays_base = np.concatenate(rays_base, 0)
    rays_base = rays_base.astype(np.float32)

    return rays_base, K_list, Rt_list, W_list, H_list