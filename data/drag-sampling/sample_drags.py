import os
from os import path as osp
from glob import glob
import json

import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import nvdiffrast.torch as dr

from render.mesh import load_mesh, batch_mesh, make_mesh
from render.render import render_mesh, interpolate
from render.light import DirectionalLight

MOVING_THRESHOLD = 0.1
VISIBLE_NUM_PIXELS_THRESHOLD = 100
SIZE_PER_MAX_DRAG_POINT = 5000

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--mesh_root", type=str, required=False)
    parser.add_argument("--num_renders", type=int, default=12)
    parser.add_argument("--render_part_mask", action="store_true")
    parser.add_argument("--sample_drags", action="store_true")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    if args.mesh_root is None:
        args.mesh_root = args.render_root
    os.makedirs(args.save_dir, exist_ok=True)
    if args.render_part_mask:
        ctx = dr.RasterizeCudaContext()

    if not osp.isdir(args.render_root) or not osp.exists(osp.join(args.render_root, "rendered.flag")):
        print(f"Render root {args.render_root} does not exist. Exiting.")
        exit(0)

    for action in os.listdir(args.render_root):
        if not osp.isdir(osp.join(args.render_root, action)):
            continue
        if len(glob(osp.join(args.save_dir, action, "*.json"))) >= args.num_samples:
            continue
        try:
            random_camera_idx = np.random.randint(args.num_renders)

            action_mesh_dir = osp.join(args.mesh_root, action)
            all_mesh_files = sorted(glob(osp.join(action_mesh_dir, "*.obj")))
            if len(all_mesh_files) == 0:
                continue

            all_verts = []
            intrinsic_path = osp.join(args.render_root, "intrinsic.npy")
            extrinsic_paths = sorted(glob(osp.join(args.render_root, "[0-9][0-9][0-9].npy")))
            assert len(extrinsic_paths) == args.num_renders

            if args.render_part_mask:
                dir = torch.FloatTensor([1, 1, 1]).to("cuda")
                light = DirectionalLight(
                    color=torch.ones(3).to("cuda")[None, None, None],
                    direction=dir[None, None, None],
                    min_kd=0.4,
                )

                all_w2c = [torch.from_numpy(np.load(extrinsic_path)).to("cuda").float() for extrinsic_path in extrinsic_paths]
                all_w2c = [torch.concat([w2c, torch.Tensor([[0, 0, 0, 1]]).to(w2c)], dim=0) for w2c in all_w2c]
                all_w2c = torch.stack(all_w2c, dim=0)
                proj = torch.from_numpy(np.load(intrinsic_path)).to("cuda").float()
                mtx_in = torch.matmul(proj[None], all_w2c)
                view_pos = torch.linalg.inv(all_w2c)[:, :3, 3]

            for i, mesh_file in enumerate(all_mesh_files):
                mesh_verts = {}
                frame_id = mesh_file.split("/")[-1].split(".")[0].split("_")[-1]

                if args.render_part_mask:
                    whole_mesh = batch_mesh(load_mesh(mesh_file))
                    whole_mesh = make_mesh(
                        verts = whole_mesh.v_pos,
                        faces = whole_mesh.t_pos_idx,
                        uvs = whole_mesh.v_tex,
                        uv_idx = whole_mesh.t_tex_idx,
                        material = whole_mesh.material,
                    )
                    with torch.no_grad():
                        out, rast_out_input = render_mesh(
                            ctx,
                            whole_mesh,
                            mtx_in=mtx_in,
                            view_pos=view_pos,
                            lgt=light,
                            resolution=(512, 512),
                            bsdf="diffuse",
                            spp=1,
                        )

                        v_pos_whole, _ = interpolate(whole_mesh.v_pos, rast_out_input, whole_mesh.t_pos_idx[0].int())

                        v_pos_whole_to_save = torch.flip(v_pos_whole, [1])
                        save_path = osp.join(args.save_dir, action, f"whole_vpos_{frame_id}.pt")
                        os.makedirs(osp.dirname(save_path), exist_ok=True)
                        torch.save(v_pos_whole_to_save.cpu(), save_path)

                with open(mesh_file, "r") as f:
                    lines = f.readlines()
                
                current_obj_name = "Obj"
                for line in lines:
                    if len(line.split()) == 0:
                        continue
                    prefix = line.split()[0].lower()
                    if prefix == "o":
                        current_obj_name = line.split()[1]
                    elif prefix == "v":
                        if current_obj_name not in mesh_verts:
                            mesh_verts[current_obj_name] = []
                        mesh_verts[current_obj_name].append([float(v) for v in line.split()[1:]])
                all_verts.append(mesh_verts)

                if args.render_part_mask:
                    num_prev_verts = 0
                    for obj_name, verts in mesh_verts.items():
                        num_verts = len(verts)
                        valid_face_index = (whole_mesh.t_pos_idx[0] >= num_prev_verts).logical_and(
                            whole_mesh.t_pos_idx[0] < num_prev_verts + num_verts
                        ).all(dim=-1)
                        part_mesh = make_mesh(
                            verts = whole_mesh.v_pos[:, num_prev_verts:num_prev_verts+num_verts],
                            faces = whole_mesh.t_pos_idx[:, valid_face_index, :] - num_prev_verts,
                            uvs = torch.zeros_like(whole_mesh.v_pos[:, :1, :2]) + 0.5,
                            uv_idx = torch.zeros_like(whole_mesh.t_pos_idx[:, valid_face_index, :]),
                            material = whole_mesh.material,
                        )

                        with torch.no_grad():
                            out, rast_out_part = render_mesh(
                                ctx,
                                part_mesh,
                                mtx_in=mtx_in,
                                view_pos=view_pos,
                                lgt=light,
                                resolution=(512, 512),
                                bsdf="diffuse",
                                spp=1,
                            )

                            v_pos_part, _ = interpolate(part_mesh.v_pos, rast_out_part, part_mesh.t_pos_idx[0].int())

                            part_foreground_img = ((v_pos_part - v_pos_whole).norm(dim=-1) < 1e-6)
                            # Save the foreground mask
                            save_path = osp.join(args.save_dir, action, f"{obj_name}_visibility_{frame_id}.pt")
                            part_foreground_img = torch.flip(part_foreground_img, [1])
                            os.makedirs(osp.dirname(save_path), exist_ok=True)
                            torch.save(part_foreground_img.cpu(), save_path)

                        num_prev_verts += num_verts
            
            if args.sample_drags:
                # Sanity check
                valid_meshes = True
                for mesh_verts in all_verts:
                    for obj_name, verts in all_verts[0].items():
                        if not (obj_name in mesh_verts and len(verts) == len(mesh_verts[obj_name])):
                            valid_meshes = False
                    for obj_name, verts in mesh_verts.items():
                        if not (obj_name in all_verts[0] and len(verts) == len(all_verts[0][obj_name])):
                            valid_meshes = False

                if not valid_meshes:
                    continue

                # Visualize.
                all_w2c = [np.load(extrinsic_path) for extrinsic_path in extrinsic_paths]
                all_w2c = [np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0) for w2c in all_w2c]
                proj = np.load(intrinsic_path)

                total_frames = len(all_mesh_files)
                if total_frames < 14:
                    continue
                for sample_id in range(args.num_samples):
                    fps = 1
                    max_start_frame = total_frames - 14 * fps
                    start_frame = np.random.randint(0, max_start_frame)
                    first_frame_idx = all_mesh_files[start_frame].split("/")[-1].split(".")[0].split("_")[-1]
                    save_path = osp.join(args.save_dir, action, f"whole_vpos_{first_frame_idx}.pt")
                    start_v_pos = torch.load(save_path)
                    start_image_paths = sorted(glob(osp.join(args.render_root, action, f"[0-9][0-9][0-9]_{first_frame_idx}.png")))
                    all_images = [cv2.imread(image_path, cv2.IMREAD_UNCHANGED) for image_path in start_image_paths]
                    all_images_tensor = torch.stack([torch.from_numpy(image[..., -1]) > 127 for image in all_images], dim=0)

                    # Sample 1 vertex from each object.
                    sampled_vert_idx, all_tracks = [], []
                    for _ in range(args.num_renders):
                        sampled_vert_idx.append({})
                        all_tracks.append([])

                    for obj_name in all_verts[start_frame].keys():
                        # compute total displacement across frames for each vertex
                        total_displacement = 0
                        for mesh_verts_prev, mesh_verts_next in zip(all_verts[start_frame:][:-1], all_verts[start_frame+1:]):
                            total_displacement = np.linalg.norm(np.array(mesh_verts_next[obj_name]) - np.array(mesh_verts_prev[obj_name]), axis=-1) + total_displacement
                        
                        save_path = osp.join(args.save_dir, action, f"{obj_name}_visibility_{first_frame_idx}.pt")
                        visible_map = torch.load(save_path)
                        visible_map = torch.logical_and(visible_map, all_images_tensor)

                        for camera_idx in range(args.num_renders):
                            cur_total_displacement = total_displacement.copy()
                            if visible_map[camera_idx].sum() < VISIBLE_NUM_PIXELS_THRESHOLD:
                                continue
                            obj_verts = np.array(all_verts[start_frame][obj_name])
                            obj_verts = np.concatenate([obj_verts, np.ones((obj_verts.shape[0], 1))], axis=-1)
                            verts_ndc = np.dot(np.dot(proj, all_w2c[camera_idx]), obj_verts.T).T
                            verts_ndc = verts_ndc / verts_ndc[:, -1][:, None]
                            verts_ndc = verts_ndc[:, :2]
                            verts_ndc = (verts_ndc + 1) / 2
                            verts_ndc[:, 1] = 1 - verts_ndc[:, 1]
                            verts_pixel = verts_ndc * np.array([all_images_tensor.shape[2], all_images_tensor.shape[1]])[None]
                            verts_pixel = verts_pixel.astype(np.int32)
                            verts_visible = visible_map[camera_idx][
                                np.clip(verts_pixel[:, 1], 0, all_images_tensor.shape[1] - 1), 
                                np.clip(verts_pixel[:, 0], 0, all_images_tensor.shape[2] - 1)
                            ]
                            verts_visible[verts_pixel[:, 0] < 0] = False
                            verts_visible[verts_pixel[:, 1] < 0] = False
                            verts_visible[verts_pixel[:, 0] >= all_images_tensor.shape[2]] = False
                            verts_visible[verts_pixel[:, 1] >= all_images_tensor.shape[1]] = False
                            cur_total_displacement[~verts_visible] = 0

                            # One more logic to take care of 1 huge part:
                            v_pos_cur_cam = start_v_pos[camera_idx]
                            verts_pos_closest_to_cam = torch.nn.functional.grid_sample(
                                v_pos_cur_cam[None].permute(0, 3, 1, 2), 
                                torch.tensor(verts_ndc * 2 - 1, dtype=torch.float32)[None, :, None], 
                                align_corners=False
                            )
                            verts_pos_closest_to_cam = verts_pos_closest_to_cam[0, :, :, 0].permute(1, 0).numpy()
                            not_occluded_flag = np.linalg.norm(verts_pos_closest_to_cam - obj_verts[:, :3], axis=-1) < 0.01

                            cur_total_displacement[~not_occluded_flag] = 0

                            if cur_total_displacement.max() > MOVING_THRESHOLD:
                                # Sample a vertex with probability proportional to displacement
                                sampled_vert_idx[camera_idx][obj_name] = np.random.choice(
                                    len(all_verts[0][obj_name]),
                                    p=cur_total_displacement/cur_total_displacement.sum(),
                                    replace=False,
                                    size=min(
                                        np.random.randint(1, max(visible_map[camera_idx].sum() // SIZE_PER_MAX_DRAG_POINT, 1) + 1), 
                                        (cur_total_displacement > 0).sum()
                                    ),
                                ).tolist()

                    for camera_idx in range(args.num_renders):
                        for obj_name, vert_indices in sampled_vert_idx[camera_idx].items():
                            for vert_idx in vert_indices:
                                sampled_verts = [mesh_verts[obj_name][vert_idx] for mesh_verts in all_verts[start_frame:start_frame+14*fps:fps]]
                                sampled_verts = np.array(sampled_verts)
                                sampled_verts = np.concatenate([sampled_verts, np.ones((sampled_verts.shape[0], 1))], axis=-1)
                                verts_ndc = np.dot(np.dot(proj, all_w2c[camera_idx]), sampled_verts.T).T
                                verts_ndc = verts_ndc / verts_ndc[:, -1][:, None]
                                verts_ndc = verts_ndc[:, :2]
                                verts_ndc = (verts_ndc + 1) / 2
                                verts_ndc[:, 1] = 1 - verts_ndc[:, 1]
                                verts_pixel = verts_ndc * np.array([all_images_tensor.shape[2], all_images_tensor.shape[1]])[None]
                                verts_pixel = verts_pixel.astype(np.int32)

                                all_tracks[camera_idx].append(verts_pixel.tolist())

                    # Save all tracks
                    save_info = {
                        "start_frame": start_frame,
                        "fps": fps,
                        "sampled_vert_idx": sampled_vert_idx,
                        "all_tracks": all_tracks,
                    }

                    with open(osp.join(args.save_dir, action, f"sample_{sample_id:03d}.json"), "w") as f:
                        json.dump(save_info, f)

                    if args.visualize and sample_id == 0:
                        camera_idx = random_camera_idx
                        image_paths = sorted(glob(osp.join(args.render_root, action, f"{camera_idx:03d}_*.png")))
                        all_images = [cv2.imread(image_path, cv2.IMREAD_UNCHANGED) for image_path in image_paths]
                        images_circled = []
                        for f in range(14):
                            image_circled = all_images[start_frame + f*fps].copy()
                            for verts_pixel in all_tracks[camera_idx]:
                                cv2.circle(image_circled, tuple((verts_pixel[f][0], verts_pixel[f][1])), 3, (0, 255, 0), -1)
                            images_circled.append(image_circled)
                        
                        # save to gif
                        save_path = osp.join(args.save_dir, action, f"sample_{sample_id:03d}.gif")
                        images_pil = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images_circled]
                        images_pil[0].save(save_path, save_all=True, append_images=images_pil[1:], duration=200, loop=0)
        
        except Exception as e:
            print(f"Error processing {args.render_root} {action}: {e}")

print("Done")
