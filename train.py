import argparse
import logging
import os
import pickle
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from os import path as osp
from pathlib import Path
from time import time

import torch
import torch.autograd
import torch.distributed as dist
from torch.utils.data import DataLoader

import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.attention_processor import XFormersAttnProcessor
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils.import_utils import is_xformers_available
from networks import AllToFirstXFormersAttnProcessor, UNetDragSpatioTemporalConditionModel
from omegaconf import OmegaConf

from dataset import DragVideoDataset
from utils import decode_latents_and_save, sample_from_noise

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

def get_generator(loader):
    while True:
        for batch in loader:
            yield batch


def check_for_nan(module, input, output):
    if torch.isnan(output).any():
        print(f"NaN detected in {module}")


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        try:
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        except Exception as e:
            print(f"Error updating EMA for {name}: {e}")
            

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    try:
        dist.destroy_process_group()
    except:
        pass


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if logging_dir is not None:
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(
    args,
    pretrained_model_name_or_local_dir: str,
    use_wandb: bool,
    model_args: dict,
    results_dir: str,
    num_steps: int,
    global_batch_size: int,
    num_workers: int,
    log_every: int,
    ckpt_every: int,
    visualize_every: int,
    learning_rate: float,
    random_seed: int = None,
    resume_checkpoint_path: str = None,
    dataset_args: dict = None,
    log_sigma_std: float = 1.6,
    log_sigma_mean: float = 1,
    num_max_drags: int = 10,

    zero_init: bool = True,
    enable_gradient_checkpointing: bool = True,
    gradient_accumulation_steps: int = 1,

    non_first_frame_weight: float = 1.0,
    weight_increasing: bool = False,
    max_grad_norm: float = 1.0,

    test_dir: str = None,
    **kwargs,
):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert is_xformers_available(), "XFormers is required for training."

    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    if random_seed is not None:
        set_seed(random_seed, device_specific=True)
    device = accelerator.device

    # Setup an experiment folder:
    if accelerator.is_main_process:
        os.makedirs(
            results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = max([
            int(existing_dir.split("/")[-1].split("-")[0]) for existing_dir in glob(f"{results_dir}/*")
        ] + [-1]) + 1
        experiment_dir = f"{results_dir}/{experiment_index:03d}"  # Create an experiment folder
        checkpoint_dir = (
            f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        )
        samples_dir = f"{experiment_dir}/samples"  # Stores samples from the model
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # Save a copy of the config file:
        OmegaConf.save(config, os.path.join(experiment_dir, "config.yaml"))

        if use_wandb:
            run = wandb.init(
                project="DragYourWay", name=osp.basename(experiment_dir),
            )
    else:
        logger = create_logger(None)
    
    train_steps = 0 if resume_checkpoint_path is None else int(resume_checkpoint_path.split('-ckpt')[0].split("/")[-1])

    model = UNetDragSpatioTemporalConditionModel.from_pretrained(
        pretrained_model_name_or_local_dir,
        subfolder="unet",
        low_cpu_mem_usage=False,
        device_map=None,
        num_drags=num_max_drags,
        **model_args,
    )

    if zero_init:
        model.zero_init()
    
    logger.info("Model loaded")    
    logger.info(f"UNet Parameters: Trainable {sum(p.numel() for p in model.parameters() if p.requires_grad):,} / {sum(p.numel() for p in model.parameters()):,}")
    model.enable_xformers_memory_efficient_attention()

    if enable_gradient_checkpointing:
        model.enable_gradient_checkpointing()

    # Set up all-to-first attention processors.
    attn_processors_dict={
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.0.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "down_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.1.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.2.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.1.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "up_blocks.3.attentions.2.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),

        "mid_block.attentions.0.transformer_blocks.0.attn1.processor": AllToFirstXFormersAttnProcessor(),
        "mid_block.attentions.0.transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
        "mid_block.attentions.0.temporal_transformer_blocks.0.attn1.processor": XFormersAttnProcessor(),
        "mid_block.attentions.0.temporal_transformer_blocks.0.attn2.processor": XFormersAttnProcessor(),
    }

    model_attn_processor_dict = model.attn_processors
    for key in model_attn_processor_dict.keys():
        if key not in attn_processors_dict:
            attn_processors_dict[key] = model_attn_processor_dict[key]
    model.set_attn_processor(attn_processors_dict)

    ema = deepcopy(model)  # Create an EMA of the model for use after training

    image_processor = VaeImageProcessor(vae_scale_factor=8)

    for k in dataset_args:
        if "roots" in k:
            dataset_args[k] = sorted(glob(dataset_args[k]))

    dataset_train = DragVideoDataset(**dataset_args)
    loader = DataLoader(
        dataset_train,
        batch_size=int(global_batch_size // accelerator.num_processes),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
    )

    if accelerator.is_main_process:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            pretrained_model_name_or_local_dir,
            subfolder="vae",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        vae.eval()

        val_generator = get_generator(val_loader)
        
        scheduler = EulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_local_dir,
            subfolder="scheduler",
        )

    opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    model, loader, opt = accelerator.prepare(model, loader, opt)

    ema = ema.to(device)
    update_ema(
        ema, model.module if accelerator.num_processes > 1 else model, decay=0
    )  # Ensure EMA is initialized with synced weights
    if resume_checkpoint_path is not None:
        try:
            accelerator.load_state(resume_checkpoint_path)
        except:
            model_state_file = Path(resume_checkpoint_path).joinpath("model.safetensors")
            if model_state_file.exists():
                from safetensors.torch import load_file
                model_state_dict = load_file(model_state_file)
            else:
                model_state_dict = torch.load(Path(resume_checkpoint_path).joinpath("model.bin"))
            model.load_state_dict(model_state_dict, strict=False)

        ema_checkpoint_path = ''.join(resume_checkpoint_path.split("ckpt")[:-1]) + "ema.pt"
        ema.load_state_dict(torch.load(ema_checkpoint_path, map_location="cpu"), strict=False)
        logger.info(f"Loaded checkpoint from {resume_checkpoint_path}")
    requires_grad(ema, False)
    ema.eval()
    
    logger.info(f"Dataset contains {len(dataset_train):,} different (category, object, action) tuples.")

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {num_steps} steps...")

    while True:
        for batch in loader:
            with accelerator.accumulate(model):
                model.train()

                sample = batch["latents"].to(device)
                cond_latent = batch["cond_latent"].to(device)
                encoder_hidden_states = batch["embedding"].to(device)
                drags = batch["drags"].to(device)
                
                batch_size = sample.shape[0]
                
                log_sigmas = torch.randn(batch_size, device=device) * log_sigma_std + log_sigma_mean
                timesteps = log_sigmas * 0.25
                sigmas = torch.exp(log_sigmas).to(sample)
                sigmas = sigmas[(...,) + (None,) * (len(sample.shape) - 1)]
    
                model_kwargs = dict(
                    image_latents=cond_latent,
                    encoder_hidden_states=encoder_hidden_states,
                    added_time_ids=torch.FloatTensor([[6, 127, 0.02] * batch_size]).to(device),
                    drags=drags,
                )

                noise = torch.randn_like(sample)
                noised_sample = sample + sigmas * noise
                
                with accelerator.autocast():
                    model_output = model(noised_sample, timesteps, **model_kwargs)
                    target = sample
                    pred = model_output * (-sigmas / (sigmas ** 2 + 1) ** 0.5) + noised_sample / (sigmas ** 2 + 1)
                    loss_weighting = 1 + 1 / sigmas ** 2
                    loss_weighting = loss_weighting.repeat(1, 14, 1, 1, 1)
                    if weight_increasing and non_first_frame_weight > 1:
                        loss_weighting = loss_weighting * torch.linspace(1, non_first_frame_weight, 14)[..., None, None, None].to(device)
                    elif non_first_frame_weight > 1:
                        loss_weighting[:, 1:] = loss_weighting[:, 1:] * non_first_frame_weight
                    loss = (loss_weighting * torch.nan_to_num((target - pred) ** 2, nan=0.0)).mean()
                    if torch.isnan(loss):
                        raise ValueError(f"NaN loss iteraton: {train_steps}")
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

                opt.step()
                opt.zero_grad(set_to_none=True)

                # Log loss values:
                running_loss += loss.item()
                log_steps += 1
                train_steps += 1
                
                if train_steps % gradient_accumulation_steps == 0:
                    update_ema(ema, model)

                log_dict = {}
                if train_steps % log_every == 0:
                    if accelerator.num_processes > 1:
                        torch.cuda.synchronize()

                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    if accelerator.num_processes > 1:
                        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / accelerator.num_processes
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                    )
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                    log_dict["train_loss"] = avg_loss
                    log_dict["train_steps_per_sec"] = steps_per_sec
                
                del loss

            # Save checkpoints.
            if train_steps % ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    accelerator.save_state(f"{checkpoint_dir}/{train_steps:07d}-ckpt")
                    torch.save(ema.state_dict(), f"{checkpoint_dir}/{train_steps:07d}-ema.pt")
                    logger.info(f"Saved checkpoint on step {train_steps}")
                if accelerator.num_processes > 1:
                    dist.barrier()
                
            if train_steps % visualize_every == 0:
                model.eval()
                if accelerator.is_main_process:
                    val_batch = next(val_generator)
                    sample_latent = sample_from_noise(
                        model, scheduler,
                        cond_latent=val_batch["cond_latent"].to(device),
                        cond_embedding=val_batch["embedding"].to(device),
                        drags=val_batch["drags"].to(device),
                        max_guidance=1,
                    )

                    sample_latent_ema = sample_from_noise(
                        ema, scheduler,
                        cond_latent=val_batch["cond_latent"].to(device),
                        cond_embedding=val_batch["embedding"].to(device),
                        drags=val_batch["drags"].to(device),
                        max_guidance=1,
                    )

                    vae = vae.to(device)
        
                    if test_dir is not None:
                        test_bid = (train_steps // visualize_every) % 100
                        val_batch_fpath = osp.join(test_dir, f"{test_bid:05d}.pkl")
                        val_batch_val = pickle.load(open(val_batch_fpath, "rb"))
                        sample_latent_val = sample_from_noise(
                            model, scheduler,
                            cond_latent=val_batch_val["cond_latent"].to(device),
                            cond_embedding=val_batch_val["embedding"].to(device),
                            drags=val_batch_val["drags"].to(device),
                            max_guidance=1,
                        )
                        decode_latents_and_save(
                            vae, image_processor, 
                            sample_latent_val[0], f"{samples_dir}/sample_{train_steps:07d}_gen_val.gif", val_batch_val["drags"][0].to(device)
                        )
                        decode_latents_and_save(
                            vae, image_processor, 
                            val_batch_val["latents"][0].to(device), f"{samples_dir}/sample_{train_steps:07d}_gt_val.gif", val_batch_val["drags"][0].to(device)
                        )
                    
                    decode_latents_and_save(
                        vae, image_processor, 
                        sample_latent[0], f"{samples_dir}/sample_{train_steps:07d}_gen.gif", val_batch["drags"][0].to(device)
                    )
                    decode_latents_and_save(
                        vae, image_processor, 
                        sample_latent_ema[0], f"{samples_dir}/sample_{train_steps:07d}_gen_ema.gif", val_batch["drags"][0].to(device)
                    )
                    decode_latents_and_save(
                        vae, image_processor, 
                        val_batch["latents"][0].to(device), f"{samples_dir}/sample_{train_steps:07d}_gt.gif", val_batch["drags"][0].to(device)
                    )
                    vae = vae.to("cpu")

                    log_dict["gt_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gt.gif")
                    log_dict["gen_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gen.gif")
                    log_dict["gen_ema_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gen_ema.gif")
                    if test_dir is not None:
                        log_dict["gt_val_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gt_val.gif")
                        log_dict["gen_val_video"] = wandb.Video(f"{samples_dir}/sample_{train_steps:07d}_gen_val.gif")

            if accelerator.is_main_process and use_wandb:
                wandb.log(log_dict)

            if train_steps >= num_steps:
                break

        if train_steps >= num_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout

    logger.info("Done!")
    if accelerator.num_processes > 1:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--local-rank", default=0, type=int, help="Local rank for distributed training"
    )

    args = parser.parse_args()
    name = Path(args.config).stem
    config = OmegaConf.load(args.config)
    main(args, use_wandb=args.wandb, **config)
