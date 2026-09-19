"""FLAME LBS helpers matching official DECA (including homogen pad of 0)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rot_mat_to_euler(rot_mats: torch.Tensor) -> torch.Tensor:
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] + rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary):
    batch, n_verts = vertices.shape[:2]
    lmk_faces = faces[lmk_faces_idx.reshape(-1)].view(batch, -1, 3)
    lmk_faces = lmk_faces + torch.arange(batch, device=vertices.device).view(-1, 1, 1) * n_verts
    lmk_vertices = vertices.reshape(-1, 3)[lmk_faces]
    return torch.einsum("blfi,blf->bli", lmk_vertices, lmk_bary)


def batch_rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    batch = rot_vecs.shape[0]
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch, 1), dtype=rot_vecs.dtype, device=rot_vecs.device)
    k = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(batch, 3, 3)
    ident = torch.eye(3, dtype=rot_vecs.dtype, device=rot_vecs.device).unsqueeze(0)
    return ident + sin * k + (1 - cos) * torch.bmm(k, k)


def lbs(betas, pose, v_template, shapedirs, posedirs, j_regressor, parents, weights):
    batch = betas.shape[0]
    v_shaped = v_template + torch.einsum("bl,mkl->bmk", betas, shapedirs)
    joints = torch.einsum("bik,ji->bjk", v_shaped, j_regressor)
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view(batch, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:] - torch.eye(3, dtype=betas.dtype, device=betas.device)).reshape(batch, -1)
    v_posed = torch.matmul(pose_feature, posedirs).view(batch, -1, 3) + v_shaped
    joints_col = joints.unsqueeze(-1)
    rel = joints_col.clone()
    rel[:, 1:] -= joints_col[:, parents[1:]]
    transforms_mat = torch.cat(
        [
            F.pad(rot_mats.reshape(-1, 3, 3), [0, 0, 0, 1]),
            F.pad(rel.reshape(-1, 3, 1), [0, 0, 0, 1], value=1),
        ],
        dim=2,
    ).reshape(-1, j_regressor.shape[0], 4, 4)
    chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        chain.append(torch.matmul(chain[int(parents[i])], transforms_mat[:, i]))
    transforms = torch.stack(chain, 1)
    joints_h = F.pad(joints_col, [0, 0, 0, 1])
    rel_t = transforms - F.pad(torch.matmul(transforms, joints_h), [3, 0, 0, 0, 0, 0, 0, 0])
    skin = torch.matmul(weights.unsqueeze(0).expand(batch, -1, -1), rel_t.reshape(batch, j_regressor.shape[0], 16))
    skin = skin.view(batch, -1, 4, 4)
    homo = torch.cat([v_posed, torch.ones(batch, v_posed.shape[1], 1, dtype=betas.dtype, device=betas.device)], 2)
    verts = torch.matmul(skin, homo.unsqueeze(-1))[:, :, :3, 0]
    return verts, joints
