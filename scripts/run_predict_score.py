import os
import torch
import clip
from PIL import Image
import numpy as np 
from tqdm import tqdm 
import argparse
import matplotlib.pyplot as plt

gen_data_path = './scripts/gen_img_data'
gen_w_path = './scripts/gen_w'
gen_score_path = './scripts/gen_score'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_text", type=str, required=True, help="a_smile_face")
    parser.add_argument("--neg_text", type=str, required=True, help="a_face_without_smile")
    args = parser.parse_args()

    positive_text = args.pos_text
    negative_text = args.neg_text

    if not os.path.exists(gen_score_path):
        os.makedirs(gen_score_path)
        print(f"Created directory: {gen_score_path}")

    model, preprocess = clip.load("ViT-B/32", device='cpu', jit=False, download_root='./scripts/')
    model = model.to(device)
    text = clip.tokenize([positive_text.replace('_', ' '), negative_text.replace('_', ' ')]).to(device)

    img_names = os.listdir(gen_data_path)
    image_files = [file_name for file_name in img_names if file_name.endswith(('.jpg', '.png'))]

    score = []

    with torch.no_grad():
        for image_file in tqdm(image_files):
            image = preprocess(Image.open(gen_data_path + "/" + image_file)).unsqueeze(0).to(device)
            logits_per_image, _ = model(image, text)    # logits_per_image.shape: [1, 2]
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()  # probs.shape: [1, 2]
            probs_np = [probs[0][0]]
            score.append(probs_np)

    score = np.concatenate(score, axis=0)

    # save histogram
    plt.hist(score, bins=100, color='blue')
    plt.title(f'{positive_text} score')
    plt.savefig(os.path.join(gen_score_path, f'{positive_text}_score_hist.png'))

    # save score
    score = score[:, None]
    np.save(os.path.join(gen_score_path, f'{positive_text}_score.npy'), score)