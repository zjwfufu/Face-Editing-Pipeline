import os
import torch
import numpy as np

# import sys
# sys.path.append('../../../')

from common.basemodel import BaseModel

class Face_Editing(BaseModel):
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # type              name                shape                       detail                                                     
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    # [input]           G                    None                       generator after PTI
    # [input]           ws                  [1, 18, 512]                optimized w+
    # [input]           dir                 [1, 18, 512]                vector to manipulate attribute
    # [input]           gamma               single value                step determined the dir
    # [output]          snap_shot       list([1024, 2048, 3]*N)         edit visualization for 0~gamma
    # -----------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, config):
        super(Face_Editing, self).__init__(config)
        cur_path            = os.path.dirname(os.path.realpath(__file__))
        self.device         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def process(self, G, ws, dir, gamma):

        gamma_list = torch.linspace(0, gamma, steps=50).to(self.device)

        snap_shot = []

        for i in range(len(gamma_list)):
            ws_ = ws + gamma_list[i] * dir
            img = G.synthesis(ws_, noise_mode='const')
            img = (img + 1) * (255/2)
            img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            snap_shot.append(img)

        return snap_shot


if __name__ == "__main__":
    import yaml
    print(os.getcwd())
    with open('../../../config/cfg.yaml', 'rb') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    from modules.models.face_generator.face_generator import Face_Generator

    fg = Face_Generator(config)

    G = fg.model

    fe = Face_Editing(config)

    ws = np.load('../../../config/ws_test.npz')['w']
    dir = np.load('../../../config/smile.npy')
    dir = np.repeat(dir[None, ...], G.num_ws, axis=1)

    ws = torch.from_numpy(ws).to('cuda')
    dir = torch.from_numpy(dir).to('cuda')

    snap_shot = fe.process(G, ws, dir, 5)

    import pdb
    pdb.set_trace()
