#!/usr/bin/env python3
"""Dump official-equivalent PyTorch FLAME verts into the baseline NPZ.

This Mac has no CUDA. Encoder codes already match ``deca_model.tar``; this script
runs the official FLAME LBS math in CPU PyTorch on those codes and writes
``verts`` / landmarks for mesh comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deca_mlx.config import default_config
from deca_mlx.flame import _load_flame_pickle, _to_np


def batch_rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    batch = rot_vecs.shape[0]
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.cos(angle).unsqueeze(1)
    sin = torch.sin(angle).unsqueeze(1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch, 1), dtype=rot_vecs.dtype)
    k = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(batch, 3, 3)
    ident = torch.eye(3, dtype=rot_vecs.dtype).unsqueeze(0)
    return ident + sin * k + (1 - cos) * torch.bmm(k, k)


def official_lbs(betas, pose, v_template, shapedirs, posedirs, j_regressor, parents, weights):
    batch = betas.shape[0]
    v_shaped = v_template + torch.einsum("bl,mkl->bmk", betas, shapedirs)
    joints = torch.einsum("bik,ji->bjk", v_shaped, j_regressor)
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view(batch, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:] - torch.eye(3, dtype=betas.dtype)).reshape(batch, -1)
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
    homo = torch.cat([v_posed, torch.ones(batch, v_posed.shape[1], 1, dtype=betas.dtype)], 2)
    return torch.matmul(skin, homo.unsqueeze(-1))[:, :, :3, 0]


def vertices2landmarks(vertices, faces, lmk_faces_idx, lmk_bary):
    batch, n_verts = vertices.shape[:2]
    lmk_faces = faces[lmk_faces_idx.reshape(-1)].view(batch, -1, 3)
    lmk_faces = lmk_faces + torch.arange(batch, device=vertices.device).view(-1, 1, 1) * n_verts
    lmk_vertices = vertices.reshape(-1, 3)[lmk_faces]
    return torch.einsum("blfi,blf->bli", lmk_vertices, lmk_bary)


def batch_orth_proj(points, camera):
    camera = camera.view(-1, 1, 3)
    translated = torch.cat([points[:, :, :2] + camera[:, :, 1:], points[:, :, 2:]], 2)
    return camera[:, :, :1] * translated


def main() -> None:
    cfg = default_config()
    baseline_path = ROOT / "tests" / "baselines" / "IMG_0392.npz"
    data = dict(np.load(baseline_path))
    flame = _load_flame_pickle(cfg.flame_model_path)
    faces = torch.from_numpy(_to_np(flame["f"], dtype=np.int32).astype(np.int64))
    v_template = torch.from_numpy(_to_np(flame["v_template"]))[None]
    shapedirs = _to_np(flame["shapedirs"])
    shapedirs = np.concatenate([shapedirs[:, :, : cfg.n_shape], shapedirs[:, :, 300 : 300 + cfg.n_exp]], 2)
    shapedirs = torch.from_numpy(shapedirs)
    posedirs = torch.from_numpy(_to_np(np.reshape(flame["posedirs"], [-1, flame["posedirs"].shape[-1]]).T))
    j_regressor = torch.from_numpy(_to_np(flame["J_regressor"]))
    parents = _to_np(flame["kintree_table"][0], dtype=np.int32)
    parents[0] = -1
    parents_t = torch.from_numpy(parents.astype(np.int64))
    weights = torch.from_numpy(_to_np(flame["weights"]))

    shape = torch.from_numpy(data["shape"])[None]
    exp = torch.from_numpy(data["exp"])[None]
    pose = torch.from_numpy(data["pose"])[None]
    cam = torch.from_numpy(data["cam"])[None]
    betas = torch.cat([shape, exp], 1)
    full_pose = torch.cat([pose[:, :3], torch.zeros(1, 3), pose[:, 3:], torch.zeros(1, 6)], 1)
    verts = official_lbs(betas, full_pose, v_template, shapedirs, posedirs, j_regressor, parents_t, weights)

    lmk = np.load(cfg.flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")[()]
    static_idx = torch.from_numpy(np.array(lmk["static_lmk_faces_idx"], dtype=np.int64))[None]
    static_b = torch.from_numpy(np.array(lmk["static_lmk_bary_coords"], dtype=np.float32))[None]
    full_idx = torch.from_numpy(np.array(lmk["full_lmk_faces_idx"], dtype=np.int64))
    full_b = torch.from_numpy(np.array(lmk["full_lmk_bary_coords"], dtype=np.float32))
    neck_chain = []
    curr = 1
    while curr != -1:
        neck_chain.append(curr)
        curr = int(parents[curr])
    aa_pose = full_pose.view(1, -1, 3)[:, neck_chain]
    rot_mats = batch_rodrigues(aa_pose.reshape(-1, 3)).view(1, -1, 3, 3)
    rel = torch.eye(3).unsqueeze(0)
    for idx in range(len(neck_chain)):
        rel = torch.bmm(rot_mats[:, idx], rel)
    sy = torch.sqrt(rel[:, 0, 0] * rel[:, 0, 0] + rel[:, 1, 0] * rel[:, 1, 0])
    y_rot = torch.round(torch.clamp(torch.atan2(-rel[:, 2, 0], sy) * 180.0 / np.pi, max=39)).long()
    neg_mask = (y_rot < 0).long()
    mask = (y_rot < -39).long()
    y_rot = neg_mask * (mask * 78 + (1 - mask) * (39 - y_rot)) + (1 - neg_mask) * y_rot
    dyn_lut_idx = np.array(lmk["dynamic_lmk_faces_idx"], dtype=np.int64)
    dyn_lut_b = np.array(lmk["dynamic_lmk_bary_coords"], dtype=np.float32)
    dyn_idx = torch.from_numpy(dyn_lut_idx[int(y_rot[0])])[None]
    dyn_b = torch.from_numpy(dyn_lut_b[int(y_rot[0])])[None]
    lmk_idx = torch.cat([dyn_idx, static_idx], 1)
    lmk_b = torch.cat([dyn_b, static_b], 1)
    landmarks2d = vertices2landmarks(verts, faces, lmk_idx, lmk_b)
    landmarks3d = vertices2landmarks(verts, faces, full_idx.repeat(1, 1), full_b.repeat(1, 1, 1))
    landmarks2d = batch_orth_proj(landmarks2d, cam)[:, :, :2]
    landmarks2d[:, :, 1:] = -landmarks2d[:, :, 1:]
    landmarks3d = batch_orth_proj(landmarks3d, cam)
    landmarks3d[:, :, 1:] = -landmarks3d[:, :, 1:]

    data["verts"] = verts[0].detach().cpu().numpy()
    data["landmarks2d"] = landmarks2d[0].detach().cpu().numpy()
    data["landmarks3d"] = landmarks3d[0].detach().cpu().numpy()
    np.savez(baseline_path, **data)
    print(f"updated {baseline_path}")
    print(f"  verts {data['verts'].shape}  landmarks2d {data['landmarks2d'].shape}")


if __name__ == "__main__":
    main()
