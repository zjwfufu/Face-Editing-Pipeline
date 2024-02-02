import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import copy
from tqdm import tqdm

# import sys
# sys.path.append('../../../')

from common.basemodel import BaseModel

class Face_Inversion(BaseModel):
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # type              name                shape                       detail                                                     
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # [input]           G                    None                       pretrained FFHQ stylegan generator from "face_generator" class
    # [input]           target_pil           (256, 256, 3)              target image for inversion, PIL format
    # [output]          proj_w              [1, 18, 512]                ws after optimization
    # [output]          new_G                None                       stylegan generator after PTI
    # [output]          w_snap_shot         list([1024, 2048, 3]*N)     optimization visualization for w, numpy array
    # [output]          pti_snap_shot       list([1024, 2048, 3]*N)     optimization visualization for G, numpy array
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, config):
        super(Face_Inversion, self).__init__(config)
        cur_path            = os.path.dirname(os.path.realpath(__file__))
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # optimize w
        self.w_num_steps                = config['models']['Face_Inversion']['w_num_steps']
        self.w_avg_samples              = config['models']['Face_Inversion']['w_avg_samples']
        self.initial_lr                 = config['models']['Face_Inversion']['initial_lr']
        self.initial_noise_factor       = config['models']['Face_Inversion']['initial_noise_factor']
        self.lr_rampdown_length         = config['models']['Face_Inversion']['lr_rampdown_length']
        self.lr_rampup_length           = config['models']['Face_Inversion']['lr_rampup_length']
        self.noise_ramp_length          = config['models']['Face_Inversion']['noise_ramp_length']
        self.regularize_noise_weight    = config['models']['Face_Inversion']['regularize_noise_weight']
        # optimize G
        self.pti_num_steps      = config['models']['Face_Inversion']['pti_num_steps']
        self.pti_lr             = config['models']['Face_Inversion']['pti_lr']
        self.pti_l2_lambda      = config['models']['Face_Inversion']['pti_l2_lambda']
        self.pti_lpips_lambda   = config['models']['Face_Inversion']['pti_lpips_lambda']
        # calculating loss
        self.network        = None
        self.weight_path    = os.path.join(cur_path, '../../weights', config['models']['Face_Inversion']['vgg16_weights'])
        self.vgg16          = self._load_model(self.network, self.weight_path)
        self.l2_criterion   = torch.nn.MSELoss(reduction='mean')
        self.transform      = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    
    def _load_model(self, network=None, weights=None):
        with open(weights, 'rb') as f:
            vgg16 = torch.jit.load(f).eval().to(self.device)
        return vgg16
    

    def preprocess(self, G, target_pil):
        w, h     = target_pil.size
        s       = min(w, h)
        target_pil      = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        target_pil      = target_pil.resize((G.img_resolution, G.img_resolution), Image.LANCZOS)
        target_uint8    = np.array(target_pil, dtype=np.uint8)
        return target_uint8


    def process(self, G, target_pil):
        
        target_uint8            =   self.preprocess(G, target_pil)

        w_avg, w_std            =   self.w_prepare(G)

        proj_w, w_snap_shot     =   self.w_projector(G, target_uint8, w_avg, w_std)

        new_G, pti_snap_shot    =   self.finetune_generator(G, target_uint8, proj_w)

        return new_G, proj_w, w_snap_shot, pti_snap_shot
    

    def w_prepare(self, G):
        G = copy.deepcopy(G).eval().requires_grad_(False).to(self.device)

        # Compute w stats.
        print(f'>>> Computing W midpoint and stddev using {self.w_avg_samples} samples...')
        z_samples = np.random.RandomState(123).randn(self.w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(self.device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]

        w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / self.w_avg_samples) ** 0.5
        print(">>> Computing W midpoint completed.")

        return w_avg, w_std
    

    def w_projector(self, G, target_uint8, w_avg, w_std):
        print(">>> Calculate W projection.")
        G = copy.deepcopy(G).eval().requires_grad_(False).to(self.device)
        target = torch.tensor(target_uint8.transpose([2, 0, 1]), device=self.device)
        # target = self.transform(target_uint8).to(self.device)
        target_images = target.unsqueeze(0).to(torch.float32)

        if target_images.shape[2] > 256:
            resized_target_images = F.interpolate(target_images, size=(256, 256), mode='area')
        target_features = self.vgg16(resized_target_images, resize_images=False, return_lpips=True)

        # Setup noise inputs.
        noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

        w_opt       = torch.tensor(w_avg, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer   = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=self.initial_lr)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        snap_shot = []

        with tqdm(total=self.w_num_steps) as pbar:
            for step in range(self.w_num_steps):
                # Learning rate schedule.
                t               = step / self.w_num_steps
                w_noise_scale   = w_std * self.initial_noise_factor * max(0.0, 1.0 - t / self.noise_ramp_length) ** 2
                lr_ramp         = min(1.0, (1.0 - t) / self.lr_rampdown_length)
                lr_ramp         = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
                lr_ramp         = lr_ramp * min(1.0, t / self.lr_rampup_length)
                lr              = self.initial_lr * lr_ramp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Synth images from opt_w.
                w_noise = torch.randn_like(w_opt) * w_noise_scale
                ws      = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
                generated_images = G.synthesis(ws, noise_mode='const')

                # loss = self.lpips_loss(generated_images, target_features=target_features)
                # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
                synth_images = (generated_images + 1) * (255 / 2)
                if synth_images.shape[2] > 256:
                    synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')            

                generated_features = self.vgg16(synth_images, resize_images=False, return_lpips=True)
                loss = (target_features - generated_features).square().sum()

                # Noise regularization.
                reg_loss = 0.0
                for v in noise_bufs.values():
                    noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                    while True:
                        reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                        reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                        if noise.shape[2] <= 8:
                            break
                        noise = F.avg_pool2d(noise, kernel_size=2)

                loss = loss + reg_loss * self.regularize_noise_weight

                # Step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if (step + 1) % 50 == 0:
                    generated_images = (generated_images + 1) * (255/2)
                    generated_images = generated_images.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    snap_shot.append(np.concatenate([target_uint8, generated_images], axis=1))          

                pbar.set_postfix(loss_lpips=loss.item())
                pbar.update(1)

                # Normalize noise.
                with torch.no_grad():
                    for buf in noise_bufs.values():
                        buf -= buf.mean()
                        buf *= buf.square().mean().rsqrt()

        return w_opt.repeat([1, G.mapping.num_ws, 1]), snap_shot
    
    def finetune_generator(self, G, target_uint8, w_pivot):
        print(">>> Finetune Generator.")
        G               = copy.deepcopy(G).train().requires_grad_(True).to(self.device)
        target          = torch.tensor(target_uint8.transpose([2, 0, 1]), device=self.device)
        # target = self.transform(target_uint8).to(self.device)
        target_images   = target.unsqueeze(0).to(torch.float32)

        optimizer = torch.optim.Adam(G.parameters(), betas=(0.9, 0.999), lr=self.pti_lr)

        snap_shot = []

        # Dry run
        z = torch.randn([1, G.z_dim]).to(self.device) 
        c = None
        _ = G(z, c)

        with tqdm(total=self.pti_num_steps) as pbar:
            for step in range(self.pti_num_steps):
                # Synth images from opt_w.
                generated_images = G.synthesis(w_pivot, noise_mode='const')
                generated_images = (generated_images + 1) * (255/2)

                loss = 0.0

                loss_l2 = self.l2_loss(generated_images, target_images)
                loss += loss_l2 * self.pti_l2_lambda

                loss_lpips = self.lpips_loss(generated_images, target_images)
                loss += loss_lpips * self.pti_lpips_lambda

                # Step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                if (step + 1) % 50 == 0:
                    synth_image = G.synthesis(w_pivot, noise_mode='const')
                    synth_image = (synth_image + 1) * (255/2)
                    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                    snap_shot.append(np.concatenate([target_uint8, synth_image], axis=1)) 
                    # print(f'optimize G step {step+1:>4d}/{self.pti_num_steps}: loss_lpips {float(loss_lpips):<5.4f}: loss_l2 {float(l2_loss_val):<5.4f}')

                pbar.set_postfix(loss_lpips=loss_lpips.item(), loss_l2=loss_l2.item())
                pbar.update(1)

        return G, snap_shot

    def lpips_loss(self, generated_images, target_images=None, target_features=None):
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if generated_images.shape[2] > 256:
            generated_images = F.interpolate(generated_images, size=(256, 256), mode='area')

        generated_features = self.vgg16(generated_images, resize_images=False, return_lpips=True)

        if target_features is not None:
            loss = (target_features - generated_features).square().sum()
        elif target_images is not None:
            if target_images.shape[2] > 256:
                target_images = F.interpolate(target_images, size=(256, 256), mode='area')
            target_features = self.vgg16(target_images, resize_images=False, return_lpips=True)
            loss = (target_features - generated_features).square().sum()

        return loss
    
    def l2_loss(self, generated_images, target_images):
        generated_images  = generated_images / 255.
        target_images = target_images / 255.
        loss = self.l2_criterion(target_images, generated_images)
        return loss  


if __name__ == "__main__":
    import yaml
    print(os.getcwd())
    with open('../../../config/cfg.yaml', 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    fi = Face_Inversion(config)

    img_path = '../../../config/test_crop.jpg'
    target_pil = Image.open(img_path).convert('RGB')

    from modules.models.face_generator.face_generator import Face_Generator

    fg = Face_Generator(config)

    G = fg.model

    new_G, proj_w, w_snap_shot, pti_snap_shot = fi.process(G, target_pil)



        