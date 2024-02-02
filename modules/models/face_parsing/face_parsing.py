import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

# import sys
# sys.path.append('../../../')

from common.basemodel import BaseModel
from modules.models.face_parsing.networks.bisenet import BiSeNet

class Face_Parsing(BaseModel):
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # type              name                shape                       detail                                                     
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # [input]           img_np              [1024, 1024, 3]            numpy array, generated from stylegan generator
    # [output]          edit_mask           [1024, 1024]               PIL format, matted mask
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, config):
        super(Face_Parsing, self).__init__(config)
        self.cur_path       = os.path.dirname(os.path.realpath(__file__))
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight_path    = os.path.join(self.cur_path, '../../weights', config['models']['Face_Parsing']['bisenet_weights'])
        self.n_classes = config['models']['Face_Parsing']['n_classes']
        self.network        = BiSeNet(n_classes=self.n_classes, config=config)
        self.model          = self._load_model(self.network, self.weight_path)

        self.transform_size = config['models']['Face_Parsing']['transform_size']
        
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    ########################## label illustration ##########################
    # 0: backgroud  1: skin     2: l_brow   3: r_brow   4: l_eye    5: r_eye
    # 6: eye_g      7: l_ear    8: r_ear    9: ear_r    10: nose    11: mouth
    # 12: u_lip     13: l_lip   14: neck    15: neck_l  16. cloth   17: hair
    # 18: hat
    ########################################################################

    def _load_model(self, network=None, weights=None):
        state_dict = torch.load(weights)
        network.load_state_dict(state_dict)
        network = network.to(self.device)
        network.eval()
        return network
    
    def process(self, img_np,  change_hair = False):
        img = self.img_transform(img_np)
        img = img[None, :, :, :]

        _, _, img_h, img_w = img.shape

        img = img.to(self.device)

        img = F.interpolate(img, size=(self.transform_size, self.transform_size), mode='area')

        # inference
        with torch.no_grad():
            out = self.network(img)[0]
        
        # resize and save mask
        out = F.interpolate(out, size=(img_h, img_w), mode='area')
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        edit_mask_np = np.zeros((parsing.shape[0], parsing.shape[1]))

        if change_hair:
            for i in range(0, self.n_classes):
                if i not in [0, 14, 15, 16, 17, 18]:
                    index = np.where(parsing == i)
                    edit_mask_np[index[0], index[1]] = 1
        else:
            for i in range(0, self.n_classes):
                if i not in [0, 14, 15, 16, 18]:
                    index = np.where(parsing == i)
                    edit_mask_np[index[0], index[1]] = 1
            

        edit_mask = Image.fromarray((edit_mask_np * 255).astype(np.uint8), mode="L")
        edit_mask = edit_mask.point(lambda p: 255 if p > 128 else 0)

        return edit_mask


if __name__ == "__main__":
    import yaml
    print(os.getcwd())
    with open('../../../config/cfg.yaml', 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    fp = Face_Parsing(config)

    img_path = '../../../config/test_crop.jpg'
    img_pil = Image.open(img_path).convert('RGB')
    img_np = np.array(img_pil)

    matte_pil = fp.process(img_np)

    import pdb
    pdb.set_trace()
