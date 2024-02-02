# Face Editing Pipeline

​	This repository serves as a straightforward pipeline for facial attribute editing. Leveraging classic techniques and off-the-shelf deep learning models, easy for beginners to learn.

![](./assets/project.svg)

## Framework

```
.
├── assets
│   └── ...
├── attributes
│   └── ...
├── common
│   ├── basemodel.py
│   ├── __init__.py
│   ├── svm_util.py
│   └── tools.py
├── config
│   └── cfg.yaml
├── scripts
│   ├── run_generate_data.py
│   ├── run_predict_score.py
│   └── run_svm.py
├── input
│   └── ...
├── modules
│   ├── models
│	│	├── face_alignment
│   │   │   ├── face_alignment.py
│   │   │   └── networks
│	│	│		└── ...
│   │   ├── face_editing
│   │   │   └── face_editing.py
│   │   ├── face_generator
│   │   │   ├── face_generator.py
│   │   │   └── networks
│	│	│		└── ...
│   │   ├── face_inversion
│   │   │   └── face_inversion.py
│   │   ├── face_parsing
│   │   │   ├── face_parsing.py
│   │   │   └── networks
				└── ...
│   │   └── face_paste
│   │       └── face_paste.py
│   └── weights
|       └── ...
├── output
│   └── ...
└── edit.py
```

## Pretrained Model

​	pretrained model are stored in `./modules/weights` with following arrangement.

```
.
├── face_alignment
│   ├── 2DFAN4-11f355bf06.pth.tar
│   └── s3fd-619a316812.pth
├── face_editing
├── face_generator
│   └── ffhq.pkl
├── face_inversion
│   └── vgg16.pt
├── face_parsing
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
└── face_paste
```

- face_alignment

Download `2DFAN4-11f355bf06.pth.tar` from [here](https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar),  `s3fd-619a316812.pth` from [here](https://drive.google.com/file/d/1IWqJUTAZCelAZrUzfU38zK_ZM25fK32S/view).

They are famous landmarks detection models.

- face_generator

Download `ffhq.pkl` from [here](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/). It based on StyleGAN2 and trained on [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset).

- face_inversion

Download `vgg16.pt` from [here](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt). It used to calculate [LPIPS loss](https://github.com/richzhang/PerceptualSimilarity).

- face_parsing

Download `79999_iter.pth` from [here](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view),  `resnet18-5c106cde.pth` from [here](https://download.pytorch.org/models/resnet18-5c106cde.pth). It based on [modified BiSeNet](https://github.com/zllrunning/face-parsing.PyTorch?tab=readme-ov-file) and trained on [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ). It is for face semantic segmentation.

## Environment

​	The environment of this project is ordinary like common deep learning project, here only list the main part of development environment. If you are missing some libraries in your environment (`opencv-python`, `scipy`), just pip it.

- Python == 3.8
- PyTorch == 1.11
- CUDA == 11.3 
- imageio-ffmpeg == 0.4.8    (for saving `.mp4`)
- ninja == 1.11.1.1   (for compiling StyleGAN ops)

## Usage

​	Run the project by follow command:

```
python edit.py --input_dir ./input --output_dir./output --dir_path ./attributes/smile.npy --config ./config/cfg.yaml --gamma 1.5 [--save_cache] [--save_media] [--change_hair] 
```

- `--input_dir`:  Path to the folder containing input images, default is `./input`.	(The width and height of the image need to be divisible by 2 due to H264 codec.)
- `--output_dir`:  Path to the folder where the results will be saved, default is `./output`.
- `--dir_path`:  Path to the directory file with specific attributes, default is `./attributes/smile.npy`.
- `--config`: Path to the configuration file, default is `./config/cfg.yaml`.
- `--gamma`:  Value for editing strength, default is `1.5`.
- `--save_cache`:  Flag to determine whether to save cache (finetuned generator, projected w, and so on) or not, options are `True` or `False`, default is `False`.
- `--save_media`:  Flag to determine whether to save `.mp4` media files or not, options are `True` or `False`, default is `False`.
- `--change_hair`: Flag to determine whether to mask hair in `Face_Parsing` or not, default is `False`. In the long hair case, this argument had better be `False`. 

## Latent Direction

​	This repo use precomputed latent direction from [generators-with-stylegan2](https://github.com/a312863063/generators-with-stylegan2/tree/master/latent_directions), thanks for their kindness sharing. Besides, in `./scripts`, this repo also provides simple scripts for calculating latent direction through SVM and [CLIP](https://openai.com/research/clip).

​	Download `ViT-B-32.pt` from [here](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt), run these following command:

```
# generate images and latent code w
python ./scripts/run_generate_data.py
# predict score for each image by CLIP
python ./scripts/run_predict_score.py --pos_text <positive prompt> --neg_text <negative prompt>
# run svm to seek boundary
python ./scripts/run_svm.py -s <predict score path>
```

## Visualization

​	Since this repository only applies and utilizes  basic techniques, there is significant room for improvement in edited results. Additionally, as most models are trained on datasets with inherent biases towards face races, not every wild image can achieve satisfactory editing outcomes.

- Smile

https://github.com/zjwfufu/Face-Editing-Pipeline/assets/85010710/f99db5a4-35e4-4914-a611-64ce9b23cca9

https://github.com/zjwfufu/Face-Editing-Pipeline/assets/85010710/b9e3594c-79af-40e3-9bcd-5a5961f150b2

- Glasses

https://github.com/zjwfufu/Face-Editing-Pipeline/assets/85010710/0752f76d-7dfc-4699-a300-de08f0df8964

https://github.com/zjwfufu/Face-Editing-Pipeline/assets/85010710/2052ecb4-8ef7-4a25-a6c8-bb8cea479882

- Age

https://github.com/zjwfufu/Face-Editing-Pipeline/assets/85010710/cf4c145c-9722-4714-9093-4aadc8441e19

https://github.com/zjwfufu/Face-Editing-Pipeline/assets/85010710/815c7e04-51f2-4ccc-ad23-563d4e20f9f2

## Reference

- [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [PTI: Pivotal Tuning for Latent-based editing of Real Images](https://github.com/danielroich/PTI)
- [FFHQ alignment](https://github.com/chi0tzp/FFHQFaceAlignment)

- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
- [generators-with-stylegan2](https://github.com/a312863063/generators-with-stylegan2/tree/master)

## Acknowledgement

Thanks DongFang Hu@OPPO for reviewing this code.
