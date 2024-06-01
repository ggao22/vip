# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import cv2
import glob
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import numpy as np
import os 

import torch 
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image  


def load_embedding(configpath, modelpath):
    modelcfg = omegaconf.OmegaConf.load(configpath)
    cleancfg = cleanup_config(modelcfg)
    rep = hydra.utils.instantiate(cleancfg)
    model = torch.nn.DataParallel(rep)
    vip_state_dict = torch.load(modelpath, map_location=torch.device(device))['vip']
    model.load_state_dict(vip_state_dict)

    transform = T.Compose([T.Resize(224),
                    T.ToTensor()])
    return model, transform 

def main(args):
    data_path = args.data_path
    produce_gif = args.produce_gif

    model, transform = load_embedding(args.cfg_path, args.model_path)
    model.to('cuda')
    model.eval()

    embedding_names = {'vip': 'VIP'}
    colors = {'vip': 'tab:blue'}
   
    os.makedirs('embedding_data', exist_ok=True)

    data = []
    for file in sorted(os.listdir(data_path)):
        img = transform(torchvision.io.read_image(file) / 255.0)
        data.append(img)
    data = torch.stack(data)
    
    print(f'Imported Images of Shape: {data.shape}')

    with torch.no_grad():
        embeddings = model(data.cuda())
        embeddings = embeddings.cpu().numpy()

    # get goal embedding
    goal_embedding = embeddings[-1]

    # compute goal embedding distance
    distances = []
    for t in range(embeddings.shape[0]):
        cur_embedding = embeddings[t]
        cur_distance = np.linalg.norm(goal_embedding-cur_embedding)
        distances.append(cur_distance)
    distances = np.array(distances)

    fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(12,6))

    # write distances
    data_dict = {'distances': distances.reshape(-1,1)}
    npz_path = 'embedding_data/distances.npz'
    np.savez(npz_path, **data_dict)

    # Plot VIP Embedding Distance and Goal Image
    ax[0].plot(np.arange(len(distances)), distances, color=colors[rep], label=embedding_names[rep], linewidth=3)
    ax[1].imshow(imgs_cur[-1].permute(1,2,0))

    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("Frame", fontsize=15)
    ax[0].set_ylabel("Embedding Distance", fontsize=15)
    ax[0].set_title(f"VIP Embedding Distance", fontsize=15)

    ax[1].set_title("Goal Frame", fontsize=15)
    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.savefig(f"embedding_data/{rep}.png",bbox_inches='tight')
    plt.close()

    ax0_xlim = ax[0].get_xlim()
    ax0_ylim = ax[0].get_ylim()
    ax1_xlim = ax[1].get_xlim()
    ax1_ylim = ax[1].get_ylim()

    def animate(i):
        for ax_subplot in ax:
            ax_subplot.clear()
        ranges = np.arange(len(distances))
        if i >= len(distances):
            i = len(distances)-1
        line1 = ax[0].plot(ranges[0:i+1], distances[0:i+1], color="tab:blue", label="image", linewidth=3)
        line2 = ax[1].imshow(imgs_cur[i].permute(1,2,0))

        ax[0].set_title(f"VIP Embedding Distance", fontsize=15)
        ax[0].legend(loc="upper right")
        ax[0].set_xlabel("Frame", fontsize=15)
        ax[0].set_ylabel("Embedding Distance", fontsize=15)

        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title("Input Video", fontsize=15)


        ax[0].set_xlim(ax0_xlim)
        ax[0].set_ylim(ax0_ylim)
        ax[1].set_xlim(ax1_xlim)
        ax[1].set_ylim(ax1_ylim)

        return line1, line2

    # Generate animated reward curve
    if produce_gif:
        ani = FuncAnimation(fig, animate, interval=20, repeat=False, frames=len(distances)+30)
        ani.save(f"embedding_data/{rep}.gif", dpi=100, writer=PillowWriter(fps=25))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model-path', type=str)
    parser.add_argument('cfg-path', type=str)
    parser.add_argument('--data_path', type=str,
                    help='string for the path to the data file.')
    parser.add_argument('--produce_gif', default=False, action="store_true",
                    help='boolean for git production')
    args = parser.parse_args()
    rep = 'vip'
    main(args, rep)