from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from deca_mlx.lbs import batch_rodrigues, blend_shapes, lbs, vertices2joints


def test_lbs_matches_pytorch_reference():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    def t_rodrigues(rot_vecs):
        batch_size = rot_vecs.shape[0]
        angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
        rot_dir = rot_vecs / angle
        cos = torch.unsqueeze(torch.cos(angle), dim=1)
        sin = torch.unsqueeze(torch.sin(angle), dim=1)
        rx, ry, rz = torch.split(rot_dir, 1, dim=1)
        zeros = torch.zeros((batch_size, 1))
        k = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
        ident = torch.eye(3).unsqueeze(0)
        return ident + sin * k + (1 - cos) * torch.bmm(k, k)

    rng = np.random.default_rng(0)
    batch, n_verts, n_joints, n_betas = 2, 40, 5, 8
    betas = rng.normal(size=(batch, n_betas)).astype(np.float32)
    shapedirs = rng.normal(size=(n_verts, 3, n_betas)).astype(np.float32)
    v_template = rng.normal(size=(batch, n_verts, 3)).astype(np.float32)
    j_regressor = rng.normal(size=(n_joints, n_verts)).astype(np.float32)
    posedirs = rng.normal(size=((n_joints - 1) * 9, n_verts * 3)).astype(np.float32)
    parents = np.array([-1, 0, 1, 2, 2], dtype=np.int32)
    weights = np.abs(rng.normal(size=(n_verts, n_joints)).astype(np.float32))
    weights /= weights.sum(1, keepdims=True)
    pose = rng.normal(size=(batch, n_joints * 3)).astype(np.float32) * 0.2

    verts_mx, _ = lbs(
        mx.array(betas),
        mx.array(pose),
        mx.array(v_template),
        mx.array(shapedirs),
        mx.array(posedirs),
        mx.array(j_regressor),
        mx.array(parents),
        mx.array(weights),
    )
    mx.eval(verts_mx)

    betas_t = torch.from_numpy(betas)
    shapedirs_t = torch.from_numpy(shapedirs)
    v_shaped = torch.from_numpy(v_template) + torch.einsum("bl,mkl->bmk", betas_t, shapedirs_t)
    joints = torch.einsum("bik,ji->bjk", v_shaped, torch.from_numpy(j_regressor))
    rot_mats = t_rodrigues(torch.from_numpy(pose).view(-1, 3)).view(batch, -1, 3, 3)
    pose_feature = (rot_mats[:, 1:] - torch.eye(3)).reshape(batch, -1)
    v_posed = torch.matmul(pose_feature, torch.from_numpy(posedirs)).view(batch, -1, 3) + v_shaped
    joints_col = joints.unsqueeze(-1)
    rel = joints_col.clone()
    rel[:, 1:] -= joints_col[:, torch.from_numpy(parents[1:]).long()]
    transforms_mat = torch.cat(
        [
            F.pad(rot_mats.reshape(-1, 3, 3), [0, 0, 0, 1]),
            F.pad(rel.reshape(-1, 3, 1), [0, 0, 0, 1], value=1),
        ],
        dim=2,
    ).reshape(-1, n_joints, 4, 4)
    chain = [transforms_mat[:, 0]]
    parents_t = torch.from_numpy(parents).long()
    for i in range(1, n_joints):
        chain.append(torch.matmul(chain[parents_t[i]], transforms_mat[:, i]))
    transforms = torch.stack(chain, 1)
    joints_h = F.pad(joints_col, [0, 0, 0, 1])
    rel_t = transforms - F.pad(torch.matmul(transforms, joints_h), [3, 0, 0, 0, 0, 0, 0, 0])
    skin = torch.matmul(
        torch.from_numpy(weights).unsqueeze(0).expand(batch, -1, -1),
        rel_t.reshape(batch, n_joints, 16),
    ).reshape(batch, -1, 4, 4)
    homo = torch.cat([v_posed, torch.ones(batch, n_verts, 1)], 2)
    verts_pt = torch.matmul(skin, homo.unsqueeze(-1))[:, :, :3, 0].numpy()

    assert np.max(np.abs(verts_pt - np.array(verts_mx))) < 1e-4
    assert np.max(np.abs(np.array(blend_shapes(mx.array(betas), mx.array(shapedirs))) - (v_shaped.numpy() - v_template))) < 1e-5
    _ = batch_rodrigues, vertices2joints
