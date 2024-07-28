"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float, azimuth: float = None) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi if azimuth is None else azimuth
    phi = None
    while phi is None or phi < np.pi / 4 or phi > np.pi / 2:
        phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 1.2,
    minz: float = -1.2,
    only_northern_hemisphere: bool = False,
    azimuth: float = None,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """
    if azimuth is not None:
        radius = np.random.uniform(radius_min, radius_max, 1)
        x, y, z = sample_point_on_sphere(radius=radius[0], azimuth=azimuth)
    else:    
        x, y, z = _sample_spherical(
            radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
        )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def get_4x4_perspective_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 4x4 perspective matrix from the given camera."""
    # Access the camera data
    cam_data = cam.data

    # Aspect ratio
    scene = bpy.context.scene
    render = scene.render
    aspect_ratio = render.resolution_x / render.resolution_y

    # Field of View (in radians)
    fovy = cam_data.angle
    fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)
    
    # Near and Far clipping planes
    near = cam_data.clip_start
    far = cam_data.clip_end
    
    # Perspective matrix components
    f = 1 / np.tan(fovy / 2)
    matrix = Matrix((
        (f / aspect_ratio, 0, 0, 0),
        (0, f, 0, 0),
        (0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)),
        (0, 0, -1, 0)
    ))

    return cam.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=render.resolution_x,
        y=render.resolution_y,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )
    
    return matrix


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        print(obj.name)
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None


def has_global_motion(thresh) -> bool:
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    return max(offset) > thresh


def has_large_scale_change(thresh) -> bool:
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    return scale > thresh


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }


def duplicate_and_apply_modifiers(frame):
    bpy.context.scene.frame_set(frame)
    bpy.ops.object.duplicate(linked=False, mode='TRANSLATION')
    duplicated_obj = bpy.context.active_object
    
    # Apply all Armature modifiers
    for modifier in duplicated_obj.modifiers:
        if modifier.type == 'ARMATURE':
            bpy.ops.object.modifier_apply(modifier=modifier.name)
    
    return duplicated_obj


def render_object(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
    render_animation: bool = False,
    max_n_frames: int = None,
    render: bool = True,
    export_mesh: bool = False,
    filter_object_with_global_motion: bool = False,
    global_motion_threshold: float = 0.5,
    filter_object_with_large_scale_change: bool = False,
    large_scale_change_threshold: float = 0.5,
    uniform_azimuth: bool = False,
    needed_actions: List[str] = None,
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    if not render and not export_mesh:
        raise ValueError("At least one of render or export_mesh must be True.")
    if not render:
        num_renders = 1

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        load_object(object_file)

    # Set up cameras
    cam = scene.objects["Camera"]
    cam.data.lens = 35
    cam.data.sensor_width = 32

    # Set up camera constraints
    # cam_constraint = cam.constraints.new(type="TRACK_TO")
    # cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    # cam_constraint.up_axis = "UP_Y"
    # empty = bpy.data.objects.new("Empty", None)
    # scene.collection.objects.link(empty)
    # cam_constraint.target = empty

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        # missing_textures = delete_missing_textures()
        missing_textures = None
    metadata["missing_textures"] = missing_textures

    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    # normalize the scene
    normalize_scene()

    # randomize the lighting
    randomize_lighting()

    if render:
        perspective_matrix = get_4x4_perspective_matrix_from_blender(bpy.data.objects["Camera"])
        perspective_matrix_path = os.path.join(output_dir, f"intrinsic.npy")
        np.save(perspective_matrix_path, perspective_matrix)

    all_animations = set()
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            action = obj.animation_data.action
            if not obj.animation_data.nla_tracks.get(action.name.split("_")[0]):
                new_track = obj.animation_data.nla_tracks.new()
                new_track.name = action.name.split("_")[0]
                new_strip = new_track.strips.new(action.name, int(action.frame_start), action)
            obj.animation_data.action = None
        if obj.animation_data and obj.animation_data.nla_tracks:
            all_animations.update([track.name for track in obj.animation_data.nla_tracks])

    print(all_animations)
    if needed_actions is not None:
        all_animations = all_animations.intersection(set(needed_actions))

    for track_name in all_animations:
        os.makedirs(os.path.join(output_dir, track_name), exist_ok=True)

    # render the images
    all_excluded = True
    for i in range(num_renders):
        # set camera
        camera = randomize_camera(
            only_northern_hemisphere=only_northern_hemisphere,
            azimuth=np.pi * 2 * i / num_renders if uniform_azimuth else None,
        )

        if render_animation:
            if not all_animations:
                break
            for track_name in all_animations.copy():
                current_animation_output_dir = os.path.join(output_dir, track_name)

                frame_start, frame_end = 10000000, -10000000
                for obj in bpy.data.objects:
                    if obj.animation_data:
                        track = obj.animation_data.nla_tracks.get(track_name)
                        if track and track.strips:
                            frame_start = min(frame_start, min([strip.frame_start for strip in track.strips]))
                            frame_end = max(frame_end, max([strip.frame_end for strip in track.strips]))
                frame_start, frame_end = int(frame_start), int(frame_end)
                if frame_end - frame_start < 10:
                    os.removedirs(current_animation_output_dir)
                    continue                    

                for obj in bpy.data.objects:
                    if obj.animation_data:
                        for obj_track in obj.animation_data.nla_tracks:
                            obj_track.mute = (obj_track.name != track_name)

                if i == 0 and (filter_object_with_global_motion or filter_object_with_large_scale_change):
                    delta = (frame_end - frame_start) // 4
                    excluded = False
                    for frame_idx in [frame_start + delta, frame_start + 2 * delta, frame_end - delta, frame_end - 1]:
                        scene.frame_set(frame_idx)
                        if filter_object_with_global_motion or filter_object_with_large_scale_change:
                            if has_global_motion(global_motion_threshold) or has_large_scale_change(large_scale_change_threshold):
                                print(has_global_motion(global_motion_threshold), has_large_scale_change(large_scale_change_threshold))
                                os.removedirs(current_animation_output_dir)
                                excluded = True
                                all_animations.remove(track_name)
                                break
                        
                    if excluded:
                        continue
                
                all_excluded = False

                if max_n_frames is not None and frame_end - frame_start > max_n_frames:
                    every_n_frames = int((frame_end - frame_start) // max_n_frames)
                else:
                    every_n_frames = 1
            
                for frame_idx in range(frame_start, frame_end, every_n_frames):
                    scene.frame_set(frame_idx)

                    if render:
                        render_path = os.path.join(current_animation_output_dir, f"{i:03d}_{frame_idx:03d}.png")
                        scene.render.filepath = render_path
                        bpy.ops.render.render(write_still=True)

                    if i == 0 and export_mesh:
                        mesh_path = os.path.join(current_animation_output_dir, f"model_{frame_idx:03d}.obj")
                        if not os.path.exists(mesh_path):
                            bpy.ops.export_scene.obj(
                                filepath=mesh_path,
                                use_triangles=True,
                                use_materials=False,
                                group_by_object=False,
                                keep_vertex_order=True,
                                axis_up="Z",
                                axis_forward="Y",
                            )

        else:
            # render the image
            if render:
                render_path = os.path.join(output_dir, f"{i:03d}.png")
                scene.render.filepath = render_path
                bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        if render:
            rt_matrix = get_3x4_RT_matrix_from_blender(camera)
            rt_matrix_path = os.path.join(output_dir, f"{i:03d}.npy")
            np.save(rt_matrix_path, rt_matrix)

    return all_excluded


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=True,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE",
        choices=["CYCLES", "BLENDER_EEVEE"],
    )
    parser.add_argument(
        "--only_northern_hemisphere",
        action="store_true",
        help="Only render the northern hemisphere of the object.",
        default=False,
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=12,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--max_n_frames",
        type=int,
        default=None,
        help="Maximum number of frames to render for animations.",
    )
    parser.add_argument(
        "--filter_object_with_global_motion",
        action="store_true",
        help="Whether to exclude objects with large global motion.",
    )
    parser.add_argument(
        "--global_motion_threshold",
        type=float,
        default=0.3,
        help="The threshold for the center of the normalized scene to be considered as moving.",
    )
    parser.add_argument(
        "--filter_object_with_large_scale_change",
        action="store_true",
        help="Whether to exclude objects with large scale changes.",
    )
    parser.add_argument(
        "--large_scale_change_threshold",
        type=float,
        default=1.5,
        help="The threshold for the scale of the normalized scene to be considered as changing.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Whether to render the animation.",
    )
    parser.add_argument(
        "--export_mesh",
        action="store_true",
        help="Whether to export the animation as a sequence of aligned OBJ files.",
    )
    parser.add_argument(
        "--uniform_azimuth",
        action="store_true",
        help="Whether to render the object from uniformly spaced azimuth angles.",
    )
    parser.add_argument(
        "--actions",
        type=str,
        nargs="+",
        default=None,
        help="If not None, only render the animations with the given names.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render

    # Set render settings
    render.engine = args.engine
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = 512
    render.resolution_y = 512
    render.resolution_percentage = 100

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.samples = 128
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 1
    scene.cycles.transparent_max_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.filter_width = 0.01
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    bpy.context.preferences.addons["cycles"].preferences.get_devices()
    bpy.context.preferences.addons[
        "cycles"
    ].preferences.compute_device_type = "CUDA"  # or "OPENCL"

    # Render the images
    all_excluded = render_object(
        object_file=args.object_path,
        num_renders=args.num_renders,
        only_northern_hemisphere=args.only_northern_hemisphere,
        output_dir=args.output_dir,
        render_animation=True,
        max_n_frames=args.max_n_frames,
        render=args.render,
        export_mesh=args.export_mesh,
        filter_object_with_global_motion=args.filter_object_with_global_motion,
        global_motion_threshold=args.global_motion_threshold,
        filter_object_with_large_scale_change=args.filter_object_with_large_scale_change,
        large_scale_change_threshold=args.large_scale_change_threshold,
        uniform_azimuth=args.uniform_azimuth,
        needed_actions=args.actions,
    )

    # Save a flag
    if all_excluded:
        with open(os.path.join(args.output_dir, "excluded.flag"), "w") as f:
            f.write("Excluded")
    else:
        with open(os.path.join(args.output_dir, "rendered.flag"), "w") as f:
            f.write("Rendered")
