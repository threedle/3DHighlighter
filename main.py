import argparse
import clip
import copy
import json
import kaolin as kal
import kaolin.ops.mesh
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torchvision

from itertools import permutations, product
from neural_highlighter import NeuralHighlighter
from Normalization import MeshNormalizer
from mesh import Mesh
from pathlib import Path
from render import Renderer
from tqdm import tqdm
from torch.autograd import grad
from torchvision import transforms
from utils import device, color_mesh


def optimize(agrs):
    # Constrain most sources of randomness
    # (some torch backwards functions within CLIP are non-determinstic)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load CLIP model 
    clip_model, preprocess = clip.load(args.clipmodel, device, jit=args.jit)

    # Adjust output resolution depending on model type 
    res = 224 
    if args.clipmodel == "ViT-L/14@336px":
        res = 336
    if args.clipmodel == "RN50x4":
        res = 288
    if args.clipmodel == "RN50x16":
        res = 384
    if args.clipmodel == "RN50x64":
        res = 448

    Path(os.path.join(args.output_dir, 'renders')).mkdir(parents=True, exist_ok=True)

    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))

    render = Renderer(dim=(args.render_res, args.render_res))
    mesh = Mesh(args.obj_path)
    MeshNormalizer(mesh)()

    # Initialize variables
    background = None
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)
    n_augs = args.n_augs
    dir = args.output_dir

    # Record command line arguments
    with open(os.path.join(dir, 'commandline_args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # CLIP and Augmentation Transforms
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    clip_transform = transforms.Compose([
        transforms.Resize((res, res)),
        clip_normalizer
    ])
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(res, scale=(1, 1)),
        transforms.RandomPerspective(fill=1, p=0.8, distortion_scale=0.5),
        clip_normalizer
    ])
    
    # MLP Settings
    mlp = NeuralHighlighter(args.depth, args.width, out_dim=args.n_classes, positional_encoding=args.positional_encoding,
                            sigma=args.sigma).to(device)
    optim = torch.optim.Adam(mlp.parameters(), args.learning_rate)

    # list of possible colors
    rgb_to_color = {(204/255, 1., 0.): "highlighter", (180/255, 180/255, 180/255): "gray"}
    color_to_rgb = {"highlighter": [204/255, 1., 0.], "gray": [180/255, 180/255, 180/255]}
    full_colors = [[204/255, 1., 0.], [180/255, 180/255, 180/255]]
    colors = torch.tensor(full_colors).to(device)
    
    
    # --- Prompt ---
    # pre-process multi_word_inputs
    args.object[0] = ' '.join(args.object[0].split('_'))
    for i in range(len(args.classes)):
        args.classes[i] = ' '.join(args.classes[i].split('_'))
    # encode prompt with CLIP
    prompt = "A 3D render of a gray {} with highlighted {}".format(args.object[0], args.classes[0])
    with torch.no_grad():
        prompt_token = clip.tokenize([prompt]).to(device)
        encoded_text = clip_model.encode_text(prompt_token)
        encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)

    vertices = copy.deepcopy(mesh.vertices)
    losses = []

    # Optimization loop
    for i in tqdm(range(args.n_iter)):
        optim.zero_grad()

        # predict highlight probabilities
        pred_class = mlp(vertices)

        # color and render mesh
        sampled_mesh = mesh
        color_mesh(pred_class, sampled_mesh, colors)
        rendered_images, elev, azim = render.render_views(sampled_mesh, num_views=args.n_views,
                                                                show=args.show,
                                                                center_azim=args.frontview_center[0],
                                                                center_elev=args.frontview_center[1],
                                                                std=args.frontview_std,
                                                                return_views=True,
                                                                lighting=True,
                                                                background=background)

        # Calculate CLIP Loss
        loss = clip_loss(args, rendered_images, encoded_text, clip_transform, augment_transform, clip_model)
        loss.backward(retain_graph=True)

        optim.step()

        # update variables + record loss
        with torch.no_grad():
            losses.append(loss.item())

        # report results
        if i % 100 == 0:
            print("Last 100 CLIP score: {}".format(np.mean(losses[-100:])))
            save_renders(dir, i, rendered_images)
            with open(os.path.join(dir, "training_info.txt"), "a") as f:
                f.write(f"For iteration {i}... Prompt: {prompt}, Last 100 avg CLIP score: {np.mean(losses[-100:])}, CLIP score {losses[-1]}\n")

    # re-initialize background color
    if args.background is not None:
        assert len(args.background) == 3
        background = torch.tensor(args.background).to(device)
    # save results
    save_final_results(args, dir, mesh, mlp, vertices, colors, render, background)

    # Save prompts
    with open(os.path.join(dir, prompt), "w") as f:
        f.write('')



# ================== HELPER FUNCTIONS =============================
def save_final_results(args, dir, mesh, mlp, vertices, colors, render, background):
    mlp.eval()
    with torch.no_grad():
        probs = mlp(vertices)
        max_idx = torch.argmax(probs, 1, keepdim=True)
        # for renders
        one_hot = torch.zeros(probs.shape).to(device)
        one_hot = one_hot.scatter_(1, max_idx, 1)
        sampled_mesh = mesh

        highlight = torch.tensor([204, 255, 0]).to(device)
        gray = torch.tensor([180, 180, 180]).to(device)
        colors = torch.stack((highlight/255, gray/255)).to(device)
        color_mesh(one_hot, sampled_mesh, colors)
        rendered_images, _, _ = render.render_views(sampled_mesh, num_views=args.n_views,
                                                                        show=args.show,
                                                                        center_azim=args.frontview_center[0],
                                                                        center_elev=args.frontview_center[1],
                                                                        std=args.frontview_std,
                                                                        return_views=True,
                                                                        lighting=True,
                                                                        background=background)
        # for mesh
        final_color = torch.zeros(vertices.shape[0], 3).to(device)
        final_color = torch.where(max_idx==0, highlight, gray)
        objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
        mesh.export(os.path.join(dir, f"{objbase}_{args.classes[0]}.ply"), extension="ply", color=final_color)
        save_renders(dir, 0, rendered_images, name='final_render.jpg')
        

def clip_loss(args, rendered_images, encoded_text, clip_transform, augment_transform, clip_model):
    if args.n_augs == 0:
        clip_image = clip_transform(rendered_images)
        encoded_renders = clip_model.encode_image(clip_image)
        encoded_renders = encoded_renders / encoded_renders.norm(dim=1, keepdim=True)
        if args.clipavg == "view":
            if encoded_text.shape[0] > 1:
                loss = -torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                torch.mean(encoded_text, dim=0), dim=0)
            else:
                loss = -torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                encoded_text)
        else:
            loss = -torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
    elif args.n_augs > 0:
        loss = 0.0
        for _ in range(args.n_augs):
            augmented_image = augment_transform(rendered_images)
            encoded_renders = clip_model.encode_image(augmented_image)
            if args.clipavg == "view":
                if encoded_text.shape[0] > 1:
                    loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                    torch.mean(encoded_text, dim=0), dim=0)
                else:
                    loss -= torch.cosine_similarity(torch.mean(encoded_renders, dim=0, keepdim=True),
                                                    encoded_text)
            else:
                loss -= torch.mean(torch.cosine_similarity(encoded_renders, encoded_text))
    return loss

def save_renders(dir, i, rendered_images, name=None):
    if name is not None:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, name))
    else:
        torchvision.utils.save_image(rendered_images, os.path.join(dir, 'renders/iter_{}.jpg'.format(i)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=0)

    # directory structure
    parser.add_argument('--obj_path', type=str, default='data/scenes/scene.obj')
    parser.add_argument('--output_dir', type=str, default='results/segment/1')

    # mesh+prompt info
    parser.add_argument('--prompt', nargs="+", default='a pig with pants')
    parser.add_argument('--object', nargs=1, default='cow')
    parser.add_argument('--classes', nargs="+", default='sphere cube')

    # render
    parser.add_argument('--background', nargs=3, type=float, default=[1., 1., 1.])
    parser.add_argument('--n_views', type=int, default=5)
    parser.add_argument('--frontview_std', type=float, default=4)
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.])
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--n_augs', type=int, default=1)
    parser.add_argument('--clipavg', type=str, default='view')
    parser.add_argument('--render_res', type=int, default=224)

    # CLIP
    parser.add_argument('--clipmodel', type=str, default='ViT-L/14')
    parser.add_argument('--jit', action="store_true")

    # network
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--positional_encoding', action='store_true')
    parser.add_argument('--sigma', type=float, default=5.0)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--n_iter', type=int, default=2500)

    args = parser.parse_args()

    optimize(args)
