# Objaverse-Animation & Objaverse-Animation-HQ

![Dataset](https://vgg-puppetmaster.github.io/resources/data.png)

### Filtering
The `objaverse-animation.json` and `objaverse-animation-HQ.json` provide the IDs of the animated 3D models in Objaverse.

### Rendering
1. The rendering script is tested on Linux with Blender 3.2.2:
```
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz && \
tar -xf blender-3.2.2-linux-x64.tar.xz && \
rm blender-3.2.2-linux-x64.tar.xz
```

2. Download the Objaverse dataset ([instructions](https://objaverse.allenai.org/)).

3. Use the following command
```
/path/to/blender-3.2.2-linux-x64/blender -noaudio -b -P blender_render_animation.py -- --object_path /path/to/glb --output_dir /path/to/output/directory --only_northern_hemisphere --engine CYCLES --num_renders 12 --max_n_frames 32 --uniform_azimuth --render
```

### Exporting Meshes
The script also allows you to export a sequence of vertex-aligned .obj meshes for each animated 3D model, which can be useful for other applications:
```
/path/to/blender-3.2.2-linux-x64/blender -noaudio -b -P blender_render_animation.py -- --object_path /path/to/glb --output_dir /path/to/output/directory --only_northern_hemisphere --engine CYCLES --max_n_frames 32 --export_mesh
```

### Sampling drags
0. Before proceeding, make sure you have the animations rendered and mesh exported. We use the mesh to sample 3D trajectories and use the camera matrices saved from rendering to project the trajectories to the image space.

1. Use the following command
```
python sample_drags.py --render_root /path/to/the/render/root --save_dir /path/to/the/dir/to/save/the/results --num_renders 12 --render_part_mask --sample_drags --num_samples 20 --visualize
```

Notes:
- The `--render_root` is the `output_dir` specified in blender rendering.
- The `--num_renders` value should be consistent with the one used in blender rendering.
- `--visualize` flag should be disabled for efficiency concern.