# Objaverse-Animation & Objaverse-Animation-HQ

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
The script also allows you to export a sequence of .obj meshes, which can be useful for other applications:
```
/path/to/blender-3.2.2-linux-x64/blender -noaudio -b -P blender_render_animation.py -- --object_path /path/to/glb --output_dir /path/to/output/directory --only_northern_hemisphere --engine CYCLES --max_n_frames 32 --export_mesh
```
