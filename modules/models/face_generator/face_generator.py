import os
import torch
import numpy as np
import pickle
import PIL.Image
from common.basemodel import BaseModel

class Face_Generator(BaseModel):
    # -----------------------------------------------------------------------------------------------------------------------------
    # type              name                shape                detail                                                     
    # -----------------------------------------------------------------------------------------------------------------------------
    # [input]           ws                  [1, 18, 512]         w plus in stylegan, 18 channels are the same           
    # [output]          gen_img             [1, 3, 1024, 1024]   generated image from stylegan, CHW, float, range [0, 1]
    # [output]          img                 (1024, 1024, 3)      PIL image, RGB format 
    # -----------------------------------------------------------------------------------------------------------------------------
    def __init__(self, config: dict={}):
        super(Face_Generator, self).__init__(config)
        self.cur_path            = os.path.dirname(os.path.realpath(__file__))
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weight_path    = os.path.join(self.cur_path, '../../weights', config['models']['Face_Generator']['model_weights'])
        self.network        = None
        self.model          = self._load_model(self.network, self.weight_path)

        # Dry run
        print(f">>> Dry run to hook custom ops")
        z = torch.randn([1, self.model .z_dim]).to(self.device) 
        c = None
        _ = self.model(z, c)

    def _load_model(self, network=None, weights=None):
        with open(weights, 'rb') as f:
            # ugly fix, for deserialize success
            import sys
            new_path = [os.path.join(self.cur_path, 'networks')]
            sys.path.extend(new_path)
            G = pickle.load(f)['G_ema'].to(self.device)
        return G
    
    def process(self, ws):
        assert ws.shape[1:] == (self.model.num_ws, self.model.w_dim)
        gen_img = self.model.synthesis(ws, noise_mode='const', force_fp32=True)
        return gen_img
    
    def postprocess(self, gen_img):
        img = (gen_img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        return img
    
    
