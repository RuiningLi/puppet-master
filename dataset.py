import json
import os
import os.path as osp
from glob import glob
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


REDUCE_DUPLICATE_DRAGS = True


class DragVideoDataset(Dataset):
    """
    An example dataset to load the drag samples together with the rendered animations. 
    Note we pre-compute the latents and embeddings for each frame and save them in latent_dist_roots and embedding_roots, respectively.
    """
    def __init__(
        self,
        latent_dist_roots: List[str],
        embedding_roots: List[str],
        drag_roots: List[str],
        num_max_drags: int = 10,
        num_drag_samples: int = 20,
        num_views: int = 12,
        num_frames: int = 14,
        sample_size: int = 256,
    ):
        """
        Args:
            latent_dist_roots: List[str]
                Path to the root of the latent distribution. Note the order of the roots must match the order of the drag_roots.
            embedding_roots: List[str]
                Path to the root of the embedding. Note the order of the roots must match the order of the drag_roots.
            drag_roots: List[str]
                Path to the root of the drag. Note the order of the roots must match the order of the latent_dist_roots and embedding_roots.
        """
        super().__init__()

        self.latent_dist_roots = latent_dist_roots
        self.embedding_roots = embedding_roots
        self.drag_roots = drag_roots
        assert len(latent_dist_roots) == len(embedding_roots) == len(drag_roots), \
            "The length of latent_dist_roots, embedding_roots, and drag_roots must be the same. Got {} {} {}".format(
                len(latent_dist_roots), len(embedding_roots), len(drag_roots))

        self.num_max_drags = num_max_drags
        self.num_drag_samples = num_drag_samples
        self.num_views = num_views
        self.num_frames = num_frames

        self.sample_size = sample_size

        self.obj_action_tuples = []
        for obj_idx, obj_drag_root in enumerate(drag_roots):
            for action in os.listdir(obj_drag_root):
                self.obj_action_tuples.append((obj_idx, action))

    def __len__(self):
        return len(self.obj_action_tuples)
    
    def get_batch(self, index):
        obj_idx, action = self.obj_action_tuples[index % len(self.obj_action_tuples)]
        sample_id = np.random.randint(self.num_drag_samples)
        sample_fpath = osp.join(self.drag_roots[obj_idx], action, f"sample_{sample_id:03d}.json")

        with open(sample_fpath, "r") as f:
            sample = json.load(f)
        assert len(sample["all_tracks"]) == self.num_views
        view_id = np.random.randint(self.num_views)

        start_frame, fps = sample["start_frame"], sample["fps"]
        drags = sample["all_tracks"][view_id]  # num_points, num_frames, 2
        if len(drags) > self.num_max_drags:
            drags = random.sample(drags, self.num_max_drags)

        # Load latents
        latents = []
        all_latents_mean_files = sorted(glob(osp.join(
            self.latent_dist_roots[obj_idx], action, f"{view_id:03d}_*_latents_{self.sample_size // 8}_mean.pt")))
        start_frame_id = all_latents_mean_files[start_frame].split("_")[-4]

        cond_latent_fpath = all_latents_mean_files[start_frame]
        cond_latent = torch.load(cond_latent_fpath, map_location="cpu")

        for frame_id in range(start_frame, start_frame + self.num_frames * fps, fps):
            latent_mean_fpath = all_latents_mean_files[frame_id]
            latent_std_fpath = latent_mean_fpath.replace("mean", "std")
            latent_mean = torch.load(latent_mean_fpath, map_location="cpu")
            latent_std = torch.load(latent_std_fpath, map_location="cpu")
            latents.append(latent_mean + latent_std * torch.randn_like(latent_mean))
        latents = torch.stack(latents)

        # Load embedding
        embedding_fpath = osp.join(
            self.embedding_roots[obj_idx], action, 
            f"{view_id:03d}_{start_frame_id}_embedding.pt"
        )
        embedding = torch.load(embedding_fpath, map_location="cpu")

        # Preprocess drags
        # 0. Sanity check
        assert all([len(drag_point) == self.num_frames for drag_point in drags])
        assert all([len(coord) == 2 for drag_point in drags for coord in drag_point])
        # 1. Normalize to [0, 1]
        if len(drags) == 0:
            drags = torch.zeros(self.num_frames, self.num_max_drags, 2)
        else:
            drags = torch.Tensor(drags).permute(1, 0, 2)  # num_frames, num_points, 2

            if REDUCE_DUPLICATE_DRAGS:
                removed_drags = []
                # 1. Remove parallel drags
                for i in range(drags.shape[1]):
                    if i in removed_drags:
                        continue
                    for j in range(i + 1, drags.shape[1]):
                        if torch.norm(drags[:, i] - drags[:, j], dim=-1).sum() <= drags.shape[0] * 20:
                            removed_drags.append(j)
                drags = torch.cat([drags[:, i:i+1] for i in range(drags.shape[1]) if i not in removed_drags], dim=1)

                removed_drags = []
                # 2. Calculate the total displacement of each drag
                displacement = torch.norm(drags[1:] - drags[:-1], dim=-1).sum(dim=0)
                max_drag_idx = displacement.argmax()
                for i in range(drags.shape[1]):
                    if i != max_drag_idx:
                        if torch.rand(1).item() > displacement[i] / displacement.max() * 2:
                            removed_drags.append(i)
                drags = torch.cat([drags[:, i:i+1] for i in range(drags.shape[1]) if i not in removed_drags], dim=1)
        
        drags = drags / 512.
        drags = torch.cat([drags[0:1].expand_as(drags), drags], dim=-1)
        drags = torch.cat([drags, torch.zeros(self.num_frames, self.num_max_drags - drags.shape[1], 4)], dim=1)

        return latents, cond_latent, embedding, drags
    
    def __getitem__(self, index):
        latents, cond_latent, embedding, drags = self.get_batch(index)
        return dict(
            latents=latents.to(dtype=torch.float32).mul_(0.18215),
            cond_latent=cond_latent.to(dtype=torch.float32),
            embedding=embedding.to(dtype=torch.float32),
            drags=drags.to(dtype=torch.float32),
        )


if __name__ == "__main__":
    dataset = DragVideoDataset(
        latent_dist_roots=sorted(glob("/scratch/shared/beegfs/jingbo/precomputed_latents_clean/*/*")),
        embedding_roots=sorted(glob("/scratch/shared/beegfs/jingbo/data/precomputed_embeddings/*/*")),
        drag_roots=sorted(glob("/scratch/shared/beegfs/ruining/data/DragYourWay/drag_samples_v2/*/*")),
    )
    print(len(dataset))
    sample = dataset[0]
    import pdb; pdb.set_trace()
    print("Done")