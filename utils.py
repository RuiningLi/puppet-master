import torch
import numpy as np
import cv2
from PIL import Image
from copy import deepcopy
from tqdm import tqdm

from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import export_to_gif


def tensor2vid(video: torch.Tensor, processor: VaeImageProcessor, output_type: str = "np"):
    batch_size = video.shape[0]
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


def decode_latents_and_save(vae, image_processor, latents, save_path, drags=None):
    with torch.no_grad():
        frames = vae.decode(latents.to(torch.float16) / 0.18215, num_frames=14).sample.float()
    frames = tensor2vid(frames[None].detach().permute(0, 2, 1, 3, 4), image_processor, output_type="pil")[0]
    
    # Add drag visualizations.
    if drags is not None:
        final_video = []
        for fid, frame in enumerate(frames):
            frame_np = np.array(frame).copy()
            for pid in range(drags.shape[1]):
                if (drags[fid, pid] != 0).any():
                    frame_np = cv2.circle(
                        frame_np, 
                        (int(drags[fid, pid, 0] * 256), int(drags[fid, pid, 1] * 256)), 
                        3, (255, 0, 0), -1
                    )
                    frame_np = cv2.circle(
                        frame_np, 
                        (int(drags[fid, pid, 2] * 256), int(drags[fid, pid, 3] * 256)), 
                        3, (0, 255, 0), -1
                    )
                    frame_np = cv2.line(
                        frame_np, 
                        (int(drags[fid, pid, 0] * 256), int(drags[fid, pid, 1] * 256)), 
                        (int(drags[fid, pid, 2] * 256), int(drags[fid, pid, 3] * 256)), 
                        (0, 0, 255), 
                        2
                    )
            final_video.append(Image.fromarray(frame_np))
    else:
        final_video = frames

    export_to_gif(final_video, save_path)


def sample_from_noise(model, scheduler, cond_latent, cond_embedding, drags,
                      min_guidance=1.0, max_guidance=3.0, num_inference_steps=25, num_frames=14):
    model.eval()

    scheduler_inference = deepcopy(scheduler)
    scheduler_inference.set_timesteps(num_inference_steps, device=cond_latent.device)
    timesteps = scheduler_inference.timesteps
    do_classifier_free_guidance = max_guidance > 1
    latents = torch.randn((1, num_frames, 4, 32, 32)).to(cond_latent) * scheduler_inference.init_noise_sigma
    guidance_scale = torch.linspace(min_guidance, max_guidance, num_frames).unsqueeze(0).to(cond_latent)[..., None, None, None]

    for i, t in tqdm(enumerate(timesteps)):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler_inference.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = model(
                latent_model_input,
                t,
                image_latents=torch.cat([cond_latent, torch.zeros_like(cond_latent)]) if do_classifier_free_guidance else cond_latent,
                encoder_hidden_states=torch.cat([cond_embedding, torch.zeros_like(cond_embedding)]) if do_classifier_free_guidance else cond_embedding,
                added_time_ids=torch.FloatTensor([[6, 127, 0.02] * 2]).to(cond_latent) if do_classifier_free_guidance else torch.FloatTensor([[6, 127, 0.02]]).to(cond_latent),
                drags=torch.cat([drags, torch.zeros_like(drags)]) if do_classifier_free_guidance else drags,
            )

        if do_classifier_free_guidance:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        
        latents = scheduler_inference.step(noise_pred, t, latents).prev_sample
    
    return latents