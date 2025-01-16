import os
from os import path as osp
from glob import glob
from tqdm import tqdm
import torch
from PIL import Image
import cv2
import argparse
import numpy as np

from diffusers import StableVideoDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor


def convert_rgba_to_rgb_with_white_bg(image_path, background_color=None, background_image_path=None):
    """
    Convert an RGBA image to an RGB image, replacing the alpha channel with a white background.

    :param image_path: str - The path to the source image with an RGBA channel.
    :return: Image - The converted PIL Image with an RGB channel.
    """
    # Read the image with the alpha channel using cv2
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # If the image has an alpha channel, process it; otherwise, raise an error
    if image is not None and image.shape[2] == 4:
        # Extract the alpha channel and create a white background of the same size as the image
        alpha_channel = image[:, :, 3]
        white_background = np.full(image[:, :, :3].shape, 255, dtype=np.uint8)

        if background_color is None and background_image_path is None:
            background = white_background
        elif background_color is not None:
            background = np.tile(background_color, image[:, :, :1].shape)
            background = background.astype(np.uint8)
        else:
            background = process_image(background_image_path, output_size=image.shape[0])

        # Use the alpha channel as a mask to blend the image with the white background
        front = image[:, :, :3] * (alpha_channel / 255)[:, :, np.newaxis]
        back = background * (1 - alpha_channel / 255)[:, :, np.newaxis]
        converted_image = np.uint8(front + back)

        # Convert the OpenCV BGR format to PIL RGB format
        converted_image_pil = Image.fromarray(converted_image[:, :, ::-1], 'RGB')
        return converted_image_pil
    else:
        return Image.open(image_path).convert("RGB")
    

def pad_image(image, target_size, padding_color=(255, 255, 255)):
    """Pad the image to the target size if it is smaller."""
    width, height = image.size
    new_width, new_height = target_size, target_size
    
    # Create a new image with the target size and a black background
    new_image = Image.new('RGB', (new_width, new_height), padding_color)
    
    # Calculate the position to paste the original image on the new image
    paste_x = (new_width - width) // 2
    paste_y = (new_height - height) // 2
    
    # Paste the original image on the new image
    new_image.paste(image, (paste_x, paste_y))
    
    return new_image

def center_crop(image, size):
    """Center crop the image to the given size."""
    width, height = image.size
    new_width, new_height = size, size
    
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2

    return image.crop((left, top, right, bottom))

def process_image(image_path, output_size=256):
    """Read an image, pad it if smaller, center crop it to the specified size, and convert to np.uint8 array."""
    # Open the image
    with Image.open(image_path) as img:
        # Ensure the image is in RGB mode
        img = img.convert('RGB')
        
        # Pad the image if it's smaller than the target size
        if img.size[0] < output_size or img.size[1] < output_size:
            img = pad_image(img, output_size)
        
        # Perform the center crop
        img_cropped = center_crop(img, output_size)
        
        # Convert to np.uint8
        img_array = np.array(img_cropped, dtype=np.uint8)
    
    return img_array

def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)

        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)

    elif output_type == "pt":
        outputs = torch.stack(outputs)

    elif not output_type == "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


batch_size = 128
device = "cuda"

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16, variant="fp16"
)
pipe.vae.to(dtype=torch.float16, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    args = parser.parse_args()

    all_jobs = []
    dataset_root = args.dataset_root
    for category in sorted(os.listdir(dataset_root)):
        for obj in tqdm(sorted(os.listdir(osp.join(dataset_root, category)))):
            if len(glob(osp.join(dataset_root, category, obj, "rendered.flag"))) == 0:
                continue
            for action in sorted(os.listdir(osp.join(dataset_root, category, obj))):
                if not osp.isdir(osp.join(dataset_root, category, obj, action)):
                    continue
                all_jobs.append((dataset_root, category, obj, action))

    for dataset_root, category, obj, action in tqdm(all_jobs[args.job_id::args.num_jobs]):
        all_image_paths = sorted(glob(osp.join(dataset_root, category, obj, action, "*.png")))
        if len(all_image_paths) == 0:
            continue
        num_computed_mean = len(glob(osp.join(args.save_dir, category, obj, action, "*_mean.pt")))
        num_computed_std = len(glob(osp.join(args.save_dir, category, obj, action, "*_std.pt")))
        if num_computed_mean == len(all_image_paths) and num_computed_std == len(all_image_paths):
            continue
        
        for i in range(0, len(all_image_paths), batch_size):
            image_paths = all_image_paths[i:i+batch_size]
            images = [convert_rgba_to_rgb_with_white_bg(image_path) for image_path in image_paths]
            with torch.no_grad():
                images = pipe.image_processor.preprocess(images, height=256, width=256).to(device=device, dtype=torch.float16)
                latent_dist = pipe.vae.encode(images).latent_dist

                for j, image_path in enumerate(image_paths):
                    latent_std = latent_dist.std[j].cpu()
                    latent_mean = latent_dist.mean[j].cpu()
                    latent_path_std = osp.join(args.save_dir, category, obj, action, image_path.split("/")[-1].replace(".png", "_latents_32_std.pt"))
                    latent_path_mean = osp.join(args.save_dir, category, obj, action, image_path.split("/")[-1].replace(".png", "_latents_32_mean.pt"))
                    os.makedirs(osp.dirname(latent_path_std), exist_ok=True)
                    torch.save(latent_std, latent_path_std)
                    torch.save(latent_mean, latent_path_mean)
