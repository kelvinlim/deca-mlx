"""Mesh export helpers ported from official DECA ``util.write_obj`` / ``upsample_mesh``."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj(obj_filename: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    verts, uvcoords, faces, uv_faces = [], [], [], []
    with open(obj_filename, "r") as handle:
        for line in handle:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "v":
                verts.append([float(x) for x in tokens[1:4]])
            elif tokens[0] == "vt":
                uvcoords.append([float(x) for x in tokens[1:3]])
            elif tokens[0] == "f":
                face_list = [part.split("/") for part in tokens[1:]]
                for props in face_list:
                    faces.append(int(props[0]))
                    if len(props) > 1 and props[1]:
                        uv_faces.append(int(props[1]))
    faces_arr = np.array(faces, dtype=np.int32).reshape(-1, 3) - 1
    uv_faces_arr = (
        np.array(uv_faces, dtype=np.int32).reshape(-1, 3) - 1 if uv_faces else np.zeros((0, 3), dtype=np.int32)
    )
    return (
        np.array(verts, dtype=np.float32),
        np.array(uvcoords, dtype=np.float32) if uvcoords else np.zeros((0, 2), dtype=np.float32),
        faces_arr,
        uv_faces_arr,
    )


def write_obj(
    obj_name: str | Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
    inverse_face_order: bool = False,
) -> Path:
    obj_path = Path(obj_name)
    if obj_path.suffix != ".obj":
        obj_path = obj_path.with_suffix(".obj")
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    faces = np.array(faces, copy=True)
    faces = faces + 1
    if inverse_face_order:
        faces = faces[:, [2, 1, 0]]
    with open(obj_path, "w") as handle:
        if colors is None:
            for vertex in vertices:
                handle.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        else:
            for vertex, color in zip(vertices, colors):
                handle.write(
                    f"v {vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n"
                )
        for face in faces:
            handle.write(f"f {face[2]} {face[1]} {face[0]}\n")
    return obj_path


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.clip(norms, 1e-8, None)


def upsample_mesh(
    vertices: np.ndarray,
    normals: np.ndarray,
    displacement_map: np.ndarray,
    texture_map: np.ndarray | None,
    dense_template: dict,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    dense_faces = dense_template["f"]
    x_coords = dense_template["x_coords"]
    y_coords = dense_template["y_coords"]
    valid_pixel_ids = dense_template["valid_pixel_ids"]
    valid_pixel_3d_faces = dense_template["valid_pixel_3d_faces"]
    valid_pixel_b_coords = dense_template["valid_pixel_b_coords"]

    pixel_3d_points = (
        vertices[valid_pixel_3d_faces[:, 0]] * valid_pixel_b_coords[:, 0:1]
        + vertices[valid_pixel_3d_faces[:, 1]] * valid_pixel_b_coords[:, 1:2]
        + vertices[valid_pixel_3d_faces[:, 2]] * valid_pixel_b_coords[:, 2:3]
    )
    pixel_3d_normals = (
        normals[valid_pixel_3d_faces[:, 0]] * valid_pixel_b_coords[:, 0:1]
        + normals[valid_pixel_3d_faces[:, 1]] * valid_pixel_b_coords[:, 1:2]
        + normals[valid_pixel_3d_faces[:, 2]] * valid_pixel_b_coords[:, 2:3]
    )
    pixel_3d_normals = pixel_3d_normals / np.clip(
        np.linalg.norm(pixel_3d_normals, axis=-1, keepdims=True), 1e-8, None
    )
    ys = y_coords[valid_pixel_ids].astype(int)
    xs = x_coords[valid_pixel_ids].astype(int)
    displacements = displacement_map[ys, xs]
    dense_vertices = pixel_3d_points + displacements[:, None] * pixel_3d_normals
    dense_colors = texture_map[ys, xs] if texture_map is not None else None
    return dense_vertices, dense_colors, dense_faces
