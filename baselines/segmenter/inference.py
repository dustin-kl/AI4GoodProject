import click
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F

import utils.torch as ptu

from data.utils import STATS
from data.ade20k import ADE20K_CATS_PATH
from data.utils import dataset_cat_description, seg_to_rgb

from model.factory import load_model
from model.utils import inference


@click.command()
@click.option("--model-path", type=str)
@click.option("--input-dir", "-i", type=str, help="folder with input images")
@click.option("--blend-dir", '-b', type=str, help="folder with blend images")
@click.option("--output-dir", "-o", type=str, help="folder with output images")
@click.option("--gpu/--cpu", default=True, is_flag=True)
@click.option("--soft_output", default=False, is_flag=True)
def main(model_path, input_dir, output_dir, blend_dir, gpu, soft_output):
    ptu.set_gpu_mode(gpu)

    model_dir = Path(model_path).parent
    model, variant = load_model(model_path)
    model.to(ptu.device)

    normalization_name = variant["dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]
    cat_names, cat_colors = dataset_cat_description(ADE20K_CATS_PATH)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    blend_dir = Path(blend_dir)

    output_dir.mkdir(exist_ok=True)
    blend_dir.mkdir(exist_ok=True)

    list_dir = list(input_dir.iterdir())
    list_dir.sort()
    for filename in tqdm(list_dir, ncols=80):
        pil_im = Image.open(filename).copy()
        im = F.pil_to_tensor(pil_im).float() / 255
        im = im[:3]
        im = F.normalize(im, normalization["mean"], normalization["std"])
        im = im.to(ptu.device).unsqueeze(0)

        im_meta = dict(flip=False)
        logits = inference(
            model,
            [im],
            [im_meta],
            ori_shape=im.shape[2:4],
            window_size=variant["inference_kwargs"]["window_size"],
            window_stride=variant["inference_kwargs"]["window_stride"],
            batch_size=2,
        )
        seg_map = logits.argmax(0, keepdim=True)
        seg_rgb = seg_to_rgb(seg_map, cat_colors)
        seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
        pil_seg = Image.fromarray(seg_rgb[0])

        # pil_seg.save(output_dir / filename.name)

        if soft_output:
            seg_map_soft = torch.softmax(logits, dim=0)
            seg_map_soft = seg_map_soft[1]
        else:
            seg_map_soft = torch.argmax(logits, dim=0)

        seg_map_soft = seg_map_soft.cpu().numpy() * 255
        seg_map_soft = (seg_map_soft).astype(np.uint8)
        seg_map_soft = Image.fromarray(seg_map_soft)
        seg_map_soft.save(output_dir / filename.name)

        # pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
        # pil_blend.save(blend_dir / filename.name)


if __name__ == "__main__":
    main()
