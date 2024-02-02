import os
import cv2
import copy
import numpy as np
import torch
import pickle
import sys

from collections import defaultdict
from tqdm import tqdm

total_num = 50000
batch_size = 10
gen_data_path = './scripts/gen_img_data'
gen_w_path = './scripts/gen_w'
ffhq_ckpt_path = './modules/weights/face_generator/ffhq.pkl'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_batch_inputs(latent_codes, batch_size):
    total_num = latent_codes.shape[0]
    for i in range(0, total_num, batch_size):
      yield latent_codes[i:i + batch_size]

if __name__ == "__main__":

    if not os.path.exists(gen_data_path):
        os.makedirs(gen_data_path)
        print(f"Created directory: {gen_data_path}")

    if not os.path.exists(gen_w_path):
        os.makedirs(gen_w_path)
        print(f"Created directory: {gen_w_path}")


    ### load ffhq GAN ###
    with open(ffhq_ckpt_path, 'rb') as f:
        new_path = ['./modules/models/face_generator/networks']
        sys.path.extend(new_path)
        G = pickle.load(f)['G_ema'].to(device)
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    z = torch.randn([1, G.z_dim]).to(device) 
    c = None
    img = G(z, c)
    print('dry run success.')
    ######

    results = defaultdict(list)

    ### sample from z
    latent_codes = np.random.randn(total_num, G.z_dim)

    pbar = tqdm(total=total_num, leave=False)
    with torch.no_grad():
        for z in get_batch_inputs(latent_codes, batch_size):
            # w = G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
            # img = G.synthesis(w, noise_mode='const', force_fp32=True)
            batch_size = z.shape[0]
            w = G.mapping(torch.from_numpy(z).to('cuda'),
                            None)

            generated= G.synthesis(w, noise_mode='const',force_fp32=True)
            imgs = generated
            imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            imgs = imgs.detach().cpu().numpy()

            for i in range(0, batch_size):
                img = imgs[i]
                save_path = os.path.join(gen_data_path, f'{pbar.n:06d}.jpg')

                # Resize the image to 512x512
                img = cv2.resize(img, (512, 512))

                cv2.imwrite(save_path, img[:, :, ::-1])
                results['w'].append(w[i, :1, :].detach().cpu().numpy())
                pbar.update(1)
        pbar.close()

    print('Saving results.')
    for key, val in results.items():
        save_path = os.path.join(gen_w_path, f'{key}.npy')
        np.save(save_path, np.concatenate(val, axis=0))
