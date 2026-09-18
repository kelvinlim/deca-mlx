"""Linear blend skinning helpers ported from official DECA ``decalib/models/lbs.py``."""

from __future__ import annotations

import mlx.core as mx


def rot_mat_to_euler(rot_mats: mx.array) -> mx.array:
    sy = mx.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return mx.arctan2(-rot_mats[:, 2, 0], sy)


def vertices2landmarks(
    vertices: mx.array,
    faces: mx.array,
    lmk_faces_idx: mx.array,
    lmk_bary_coords: mx.array,
) -> mx.array:
    batch_size, num_verts = vertices.shape[:2]
    lmk_faces = mx.take(faces, lmk_faces_idx.reshape((-1,)), axis=0)
    lmk_faces = lmk_faces.reshape((batch_size, -1, 3))
    offsets = mx.arange(batch_size).reshape((-1, 1, 1)) * num_verts
    lmk_faces = lmk_faces + offsets
    flat_verts = vertices.reshape((-1, 3))
    lmk_vertices = mx.take(flat_verts, lmk_faces, axis=0)
    return mx.sum(lmk_vertices * lmk_bary_coords[..., None], axis=2)


def vertices2joints(j_regressor: mx.array, vertices: mx.array) -> mx.array:
    # vertices: B x V x 3, J_regressor: J x V -> B x J x 3
    return mx.einsum("bik,ji->bjk", vertices, j_regressor)


def blend_shapes(betas: mx.array, shape_disps: mx.array) -> mx.array:
    # betas: B x NB, shape_disps: V x 3 x NB -> B x V x 3
    return mx.einsum("bl,mkl->bmk", betas, shape_disps)


def batch_rodrigues(rot_vecs: mx.array, epsilon: float = 1e-8) -> mx.array:
    batch_size = rot_vecs.shape[0]
    angle = mx.linalg.norm(rot_vecs + 1e-8, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle
    cos = mx.cos(angle)[:, None, :]
    sin = mx.sin(angle)[:, None, :]
    rx = rot_dir[:, 0:1]
    ry = rot_dir[:, 1:2]
    rz = rot_dir[:, 2:3]
    zeros = mx.zeros((batch_size, 1))
    k = mx.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1).reshape((batch_size, 3, 3))
    ident = mx.eye(3)[None, :, :]
    return ident + sin * k + (1 - cos) * (k @ k)


def transform_mat(r: mx.array, t: mx.array) -> mx.array:
    # R: B x 3 x 3, t: B x 3 x 1 -> B x 4 x 4
    batch = r.shape[0]
    bottom = mx.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 1, 4))
    bottom = mx.broadcast_to(bottom, (batch, 1, 4))
    rt = mx.concatenate([r, t], axis=2)
    return mx.concatenate([rt, bottom], axis=1)


def batch_rigid_transform(rot_mats: mx.array, joints: mx.array, parents: mx.array) -> tuple[mx.array, mx.array]:
    joints = joints[..., None]
    rel_joints = mx.array(joints)
    rel_joints_tail = rel_joints[:, 1:] - mx.take(joints, parents[1:], axis=1)
    rel_joints = mx.concatenate([rel_joints[:, :1], rel_joints_tail], axis=1)

    n_joints = joints.shape[1]
    transforms_mat = transform_mat(rot_mats.reshape((-1, 3, 3)), rel_joints.reshape((-1, 3, 1)))
    transforms_mat = transforms_mat.reshape((-1, n_joints, 4, 4))

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, n_joints):
        parent = int(parents[i].item())
        transform_chain.append(transform_chain[parent] @ transforms_mat[:, i])
    transforms = mx.stack(transform_chain, axis=1)
    posed_joints = transforms[:, :, :3, 3]

    # Official F.pad(joints, [0,0,0,1]) appends a 0, not a 1.
    joints_homogen = mx.concatenate([joints, mx.zeros_like(joints[:, :, :1, :])], axis=2)
    posed_rest = transforms @ joints_homogen
    pad = mx.zeros((transforms.shape[0], n_joints, 4, 3))
    rel_transforms = transforms - mx.concatenate([pad, posed_rest], axis=3)
    return posed_joints, rel_transforms


def lbs(
    betas: mx.array,
    pose: mx.array,
    v_template: mx.array,
    shapedirs: mx.array,
    posedirs: mx.array,
    j_regressor: mx.array,
    parents: mx.array,
    lbs_weights: mx.array,
) -> tuple[mx.array, mx.array]:
    batch_size = betas.shape[0]
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    joints = vertices2joints(j_regressor, v_shaped)

    rot_mats = batch_rodrigues(pose.reshape((-1, 3))).reshape((batch_size, -1, 3, 3))
    ident = mx.eye(3)
    pose_feature = (rot_mats[:, 1:, :, :] - ident).reshape((batch_size, -1))
    pose_offsets = (pose_feature @ posedirs).reshape((batch_size, -1, 3))
    v_posed = pose_offsets + v_shaped

    _, a = batch_rigid_transform(rot_mats, joints, parents)
    num_joints = j_regressor.shape[0]
    w = mx.broadcast_to(lbs_weights[None, ...], (batch_size, lbs_weights.shape[0], lbs_weights.shape[1]))
    t = (w @ a.reshape((batch_size, num_joints, 16))).reshape((batch_size, -1, 4, 4))

    homogen = mx.ones((batch_size, v_posed.shape[1], 1))
    v_posed_homo = mx.concatenate([v_posed, homogen], axis=2)
    v_homo = t @ v_posed_homo[..., None]
    verts = v_homo[:, :, :3, 0]
    return verts, joints
