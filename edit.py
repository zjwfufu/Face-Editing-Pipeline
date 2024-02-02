import torch
import numpy as np
import json
import argparse
import yaml
import cv2
import os
from PIL import Image

from modules.models.face_alignment.face_alignment       import Face_Alignment
from modules.models.face_generator.face_generator       import Face_Generator
from modules.models.face_inversion.face_inversion       import Face_Inversion
from modules.models.face_editing.face_editing           import Face_Editing
from modules.models.face_parsing.face_parsing           import Face_Parsing
from modules.models.face_paste.face_paste               import Face_Paste

from common.tools import get_file_list, save_snap_shot, save_transition_video


class FaceEditor(object):
    def __init__(self, config=None, user_config: dict={}):
        self.config = config
        if config is None:
            print(">>> config is missing!!!")
            return
        self.init_all_models(config)
        # print(f">>> main -> config: \n{json.dumps(config, indent=4)}")
        self.init_parse_envs(user_config)

        self.device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.G_num_ws   = self.face_generator.model.num_ws

        latent_dir = np.load(self.dir_path)
        if latent_dir.shape[0] == self.G_num_ws:
            latent_dir = latent_dir[None, ...]
        else:
            latent_dir = np.repeat(latent_dir[None, ...], self.G_num_ws, axis=1)

        self.latent_dir = torch.from_numpy(latent_dir).to(self.device)

        latent_name         = os.path.basename(self.dir_path)
        self.latent_name    = os.path.splitext(latent_name)[0]
        self.gamma_name     = str(self.gamma)


    def init_all_models(self, config):
        self.face_alignment     = Face_Alignment(config)
        self.face_generator     = Face_Generator(config)
        self.face_inversion     = Face_Inversion(config)
        self.face_editing       = Face_Editing(config)
        self.face_parsing       = Face_Parsing(config)
        self.face_paste         = Face_Paste(config)

    def init_parse_envs(self, user_config):
        self.input_dir          = user_config['input_dir']
        self.out_dir            = user_config['output_dir']
        self.dir_path           = user_config['dir_path']
        self.gamma              = user_config['gamma']
        self.save_cache         = user_config['save_cache']
        self.save_media         = user_config['save_media']
        self.change_hair        = user_config['change_hair']

        if self.save_cache:
            self.save_media = True
        
    def register_path(self, img_outdir, img_name):
        img_align_path          = f"{img_outdir}/{img_name}_align.png"
        quad_path               = f"{img_outdir}/{img_name}_quad.npy"
        newG_path               = f"{img_outdir}/{img_name}_newG.pt"
        proj_w_path             = f"{img_outdir}/{img_name}_w.npz"
        w_snap_shot_path        = f"{img_outdir}/{img_name}_w_traj.mp4"
        pti_snap_shot_path      = f"{img_outdir}/{img_name}_pti_traj.mp4"
        edit_snap_shot_path     = f"{img_outdir}/{img_name}_{self.gamma_name}_edit_traj.mp4"
        transition_video_path   = f"{img_outdir}/{img_name}_{self.gamma_name}_transition.mp4"

        path_dict = {"img_align":           img_align_path,
                     "quad":                quad_path,
                     "new_G":               newG_path,
                     "proj_w":              proj_w_path,
                     "w_snap_shot":         w_snap_shot_path,
                     "pti_snap_shot":       pti_snap_shot_path,
                     "edit_snap_shot":      edit_snap_shot_path,
                     "transition_video":    transition_video_path}

        return path_dict

    def run(self):
        image_files_list = get_file_list(self.input_dir)

        for img_file in image_files_list:

            # 1. prepare save path & read img file
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            print(f">>> Synthesis: {img_name}   attribute: {self.latent_name}   gamma: {self.gamma_name}")

            img_outdir = os.path.join(self.out_dir, '_'.join([img_name, self.latent_name]))
            path = self.register_path(img_outdir, img_name)
            os.makedirs(img_outdir, exist_ok=True)

            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')

            # 2. align img
            if os.path.exists(path['img_align']) and os.path.exists(path['quad']):
                print(f">>> Found face align cache ")
                img_align = Image.open(path['img_align'])
                quad = np.load(path['quad'])
            else:
                print(f">>> Execute face align")
                img_align, quad = self.face_alignment.process(img)
                if self.save_cache:
                    img_align.save(path['img_align'])
                    np.save(path['quad'], quad)

            # 3. do GAN inversion
            if os.path.exists(path['new_G']) and os.path.exists(path['proj_w']):
                print(f">>> Found face inversion cache ")
                with open(path['new_G'], 'rb') as f:
                    new_G = torch.load(f, map_location='cpu').to(self.device)
                proj_w = np.load(path['proj_w'])['w']
                proj_w = torch.from_numpy(proj_w).to(self.device)
            else:
                print(f">>> Execute face inversion")
                new_G, proj_w, w_snap_shot, pti_snap_shot = self.face_inversion.process(self.face_generator.model,
                                                                                img_align)
                if self.save_cache:
                    np.savez(path['proj_w'], w=proj_w.detach().cpu().numpy())
                    torch.save(new_G, path['new_G'])
                
                if self.save_media:
                    save_snap_shot(w_snap_shot, path['w_snap_shot'])
                    save_snap_shot(pti_snap_shot, path['pti_snap_shot'])

            # 4. do attribute editing
            print(f">>> Execute face editing")
            edit_snap_shot = self.face_editing.process(new_G, proj_w, self.latent_dir, self.gamma)
            if self.save_media:
                save_snap_shot(edit_snap_shot, path['edit_snap_shot'])

            # 5. obtain face mask for preventing background change
            print(f">>> Execute face parsing")
            edit_img_np = edit_snap_shot[-1]
            edit_mask = self.face_parsing.process(edit_img_np,  change_hair = self.change_hair)
            orig_mask = self.face_parsing.process(img_align,    change_hair = self.change_hair)

            # 6. recover the whole edited image
            print(f">>> Execute face paste")
            edit_img_pil = Image.fromarray(edit_img_np)
            orig_img_pil = Image.fromarray(img)
            paste_img_pil = self.face_paste.process(orig_img_pil.copy(), edit_img_pil, edit_mask, orig_mask, quad)
            if self.save_media:
                save_transition_video(orig_img_pil, paste_img_pil, path['transition_video'])

            # 7. save the result
            paste_img_pil.save(f"{img_outdir}/{img_name}.png")
            print(f">>> Synthesis {img_name}    done ")

        

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    user_config = {"input_dir":     args.input_dir,
                   "output_dir":    args.output_dir,
                   "dir_path":      args.dir_path,
                   "gamma":         args.gamma,
                   "save_cache":    args.save_cache,
                   "save_media":    args.save_media,
                   "change_hair":   args.change_hair}
    
    face_editor = FaceEditor(config=config, user_config=user_config)
    face_editor.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",      default='./input',                              type=str)
    parser.add_argument("--output_dir",     default='./output',                             type=str)
    parser.add_argument("--dir_path",       required=True,                                  type=str)
    parser.add_argument("--gamma",          default=3.,                                     type=float)
    parser.add_argument('--config',         default='./config/cfg.yaml',                    type=str)
    parser.add_argument('--save_cache',     action='store_true')
    parser.add_argument('--save_media',     action='store_true')
    parser.add_argument('--change_hair',    action='store_false')
    args = parser.parse_args()
    main(args)