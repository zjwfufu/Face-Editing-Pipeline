import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageChops
import cv2

from common.basemodel import BaseModel

class Face_Paste(BaseModel):
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # type              name                shape                       detail                                                     
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # [input]           orig_img            (H, W, 3)                   original image, PIL format, arbitrary shape
    # [input]           edit_img            (1024, 1024, 3)             edited aligned img from styleGAN, PIL format
    # [input]           edit_mask           (1024, 1024)                face mask from edit_img, generated from face parsing, PIL format
    # [input]           orig_mask           (1024, 1024)                face mask from orig_img, do logical or with edit_mask, PIL format
    # [input]           quad                [4, 2]                      numpy array, store the crop coords
    # [output]          paste_img           (H, W, 3)                   final edited image
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, config):
        super(Face_Paste, self).__init__(config)
        cur_path            = os.path.dirname(os.path.realpath(__file__))
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def process(self, orig_img, edit_img, edit_mask, orig_mask, quad):

        edit_mask_np = np.array(edit_mask)
        orig_mask_np = np.array(orig_mask.resize(edit_mask.size))

        union_mask_np = np.logical_or(edit_mask_np, orig_mask_np)
        union_mask = Image.fromarray(union_mask_np.astype(np.uint8) * 255)

        edit_img.paste(edit_mask, (0, 0), mask=ImageOps.invert(union_mask))

        min_x, min_y = np.min(quad, axis=0)
        max_x, max_y = np.max(quad, axis=0)

        width = max_x - min_x
        height = max_y - min_y

        src_points = np.array([[0, 0], [0, 1024], [1024, 1024], [1024, 0]])
        dst_points = quad - np.min(quad, 0)

        M, _ = cv2.estimateAffine2D(dst_points, src_points)
        M = M.flatten()

        edit_img = edit_img.transform((int(width), int(height)), Image.AFFINE, M)

        x, y = np.min(quad, 0)

        mask = edit_img.point(lambda p : 255 if p == 0 else 0).convert("L")

        orig_img.paste(edit_img, (int(x), int(y)), mask=ImageOps.invert(mask))

        return orig_img
 