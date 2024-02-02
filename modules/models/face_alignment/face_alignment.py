import os
import torch
import numpy as np
from PIL import Image
import scipy

# import sys
# sys.path.append('../../../')

from common.basemodel import BaseModel
from modules.models.face_alignment.networks.landmarks_pytorch import LandmarksEstimation


    
class Face_Alignment(BaseModel):
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # type              name                shape                detail                                                     
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # [input]           raw_img            [H, W, 3]             input image, numpy array, arbitrary shape
    # [output]          img                (256, 256, 3)         aligned image, PIL format, 256x256
    # [output]          quad_initial       [4, 2]                used for affine transformation, need them to paste edited img back to the original image
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, config):
        super(Face_Alignment, self).__init__(config)
        cur_path            = os.path.dirname(os.path.realpath(__file__))
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.face_landmark  = LandmarksEstimation(cur_path, config, type='2D')
        self.transform_size = config['models']['Face_Alignment']['transform_size']

    def process(self, raw_img):
        img = Image.fromarray(raw_img)
        img_tensor = torch.tensor(np.transpose(raw_img, (2, 0, 1))).float().cuda()
        with torch.no_grad():
            landmarks, _ = self.face_landmark.detect_landmarks(img_tensor.unsqueeze(0),detected_faces=None)
            lm = np.asarray(landmarks[0].detach().cpu().numpy())

        # lm_chin             = lm[0: 17]            # left-right
        # lm_eyebrow_left     = lm[17: 22]           # left-right
        # lm_eyebrow_right    = lm[22: 27]           # left-right
        # lm_nose             = lm[27: 31]           # top-down
        # lm_nostrils         = lm[31: 36]           # top-down
        lm_eye_left         = lm[36: 42]           # left-clockwise
        lm_eye_right        = lm[42: 48]           # left-clockwise
        lm_mouth_outer      = lm[48: 60]           # left-clockwise
        # lm_mouth_inner      = lm[60: 68]           # left-clockwise

        # Calculate auxiliary vectors
        eye_left            = np.mean(lm_eye_left, axis=0)
        eye_right           = np.mean(lm_eye_right, axis=0)
        eye_avg             = (eye_left + eye_right) * 0.5
        eye_to_eye          = (eye_right - eye_left)
        mouth_left          = lm_mouth_outer[0]
        mouth_right         = lm_mouth_outer[6]
        mouth_avg           = (mouth_left + mouth_right) * 0.5
        eye_to_mouth        = (mouth_avg - eye_avg)

        # Choose oriented crop rectangle
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # we need save the initial quad for recovering edited alignmented images
        quad_inital = quad.copy()

        shrink = int(np.floor(qsize / self.transform_size * 0.5))
        if shrink > 1:
            rsize   = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img     = img.resize(rsize, Image.Resampling.LANCZOS)
            quad    /= shrink
            qsize   /= shrink

        # Crop
        border = max(int(np.rint(qsize * 0.1)), 3)

        crop = (int(np.floor(min(quad[:, 0]))),
                int(np.floor(min(quad[:, 1]))),
                int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        
        crop = (max(crop[0] - border, 0),
                max(crop[1] - border, 0),
                min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))
        
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad
        pad = (int(np.floor(min(quad[:, 0]))),
               int(np.floor(min(quad[:, 1]))),
               int(np.ceil(max(quad[:, 0]))),
               int(np.ceil(max(quad[:, 1]))))
        pad = (max(-pad[0] + border, 0),
               max(-pad[1] + border, 0),
               max(pad[2] - img.size[0] + border, 0),
               max(pad[3] - img.size[1] + border, 0))

        if max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            # mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
            #                   1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / (pad[0] + 1e-12), np.float32(w - 1 - x) / (pad[2] + 1e-12)),
                            1.0 - np.minimum(np.float32(y) / (pad[1] + 1e-12), np.float32(h - 1 - y) / (pad[3] + 1e-12)))

            blur = qsize * 0.01
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
            img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')

            quad += pad[:2]

        # Transform
        img = img.transform((self.transform_size, self.transform_size), Image.Transform.QUAD, (quad + 0.5).flatten(),
                            Image.Resampling.BILINEAR)

        return img, quad_inital
    

# if __name__ == "__main__":
#     import yaml
#     print(os.getcwd())
#     with open('../../../config/cfg.yaml', 'rb') as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)

#     import cv2
#     fa = Face_Alignment(config)
#     img_path = '../../../config/test.png'
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8')
#     print(img.shape)
#     img, quad_inital = fa.process(img)
#     print(img.size)
#     print(quad_inital.shape)