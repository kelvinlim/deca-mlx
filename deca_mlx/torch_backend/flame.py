"""FLAME decoder on PyTorch, matching official DECA landmark / LBS math."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from ..config import DECAConfig
from ..flame_io import flame_to_numpy as _to_np
from ..flame_io import load_flame_pickle
from .lbs import batch_rodrigues, lbs, rot_mat_to_euler, vertices2landmarks


class FLAME(nn.Module):
    def __init__(self, config: DECAConfig):
        super().__init__()
        if not config.flame_model_path.exists():
            raise FileNotFoundError(
                f"FLAME model not found at {config.flame_model_path}. "
                "Register at https://flame.is.tue.mpg.de/ and place generic_model.pkl in data/."
            )
        flame_model = SimpleNamespace(**load_flame_pickle(config.flame_model_path))
        faces = _to_np(flame_model.f, dtype=np.int32).astype(np.int64)
        v_template = _to_np(flame_model.v_template)
        shapedirs = _to_np(flame_model.shapedirs)
        shapedirs = np.concatenate(
            [shapedirs[:, :, : config.n_shape], shapedirs[:, :, 300 : 300 + config.n_exp]],
            axis=2,
        )
        posedirs = _to_np(np.reshape(flame_model.posedirs, [-1, flame_model.posedirs.shape[-1]]).T)
        j_regressor = _to_np(flame_model.J_regressor)
        parents = _to_np(flame_model.kintree_table[0], dtype=np.int32)
        parents[0] = -1
        lbs_weights = _to_np(flame_model.weights)

        self.register_buffer("faces_tensor", torch.from_numpy(faces))
        self.register_buffer("v_template", torch.from_numpy(v_template))
        self.register_buffer("shapedirs", torch.from_numpy(shapedirs))
        self.register_buffer("posedirs", torch.from_numpy(posedirs))
        self.register_buffer("J_regressor", torch.from_numpy(j_regressor))
        self.register_buffer("parents", torch.from_numpy(parents.astype(np.int64)))
        self.register_buffer("lbs_weights", torch.from_numpy(lbs_weights))
        self.register_buffer("eye_pose", torch.zeros(1, 6))
        self.register_buffer("neck_pose", torch.zeros(1, 3))

        lmk = np.load(config.flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")[()]
        self.register_buffer("lmk_faces_idx", torch.from_numpy(np.array(lmk["static_lmk_faces_idx"], dtype=np.int64)))
        self.register_buffer("lmk_bary_coords", torch.from_numpy(np.array(lmk["static_lmk_bary_coords"], dtype=np.float32)))
        self.register_buffer(
            "dynamic_lmk_faces_idx", torch.from_numpy(np.array(lmk["dynamic_lmk_faces_idx"], dtype=np.int64))
        )
        self.register_buffer(
            "dynamic_lmk_bary_coords", torch.from_numpy(np.array(lmk["dynamic_lmk_bary_coords"], dtype=np.float32))
        )
        full_idx = np.array(lmk["full_lmk_faces_idx"], dtype=np.int64)
        full_b = np.array(lmk["full_lmk_bary_coords"], dtype=np.float32)
        if full_idx.ndim > 1:
            full_idx = full_idx.reshape(-1)
        if full_b.ndim == 3:
            full_b = full_b.reshape(-1, 3)
        self.register_buffer("full_lmk_faces_idx", torch.from_numpy(full_idx))
        self.register_buffer("full_lmk_bary_coords", torch.from_numpy(full_b))

        neck_chain = []
        curr = 1
        while curr != -1:
            neck_chain.append(curr)
            curr = int(parents[curr])
        self.register_buffer("neck_kin_chain", torch.tensor(neck_chain, dtype=torch.long))

    def _find_dynamic_lmk_idx_and_bcoords(self, pose: torch.Tensor):
        batch_size = pose.shape[0]
        aa_pose = pose.view(batch_size, -1, 3)[:, self.neck_kin_chain]
        rot_mats = batch_rodrigues(aa_pose.reshape(-1, 3)).view(batch_size, -1, 3, 3)
        rel_rot_mat = torch.eye(3, dtype=pose.dtype, device=pose.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
        for idx in range(self.neck_kin_chain.shape[0]):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)
        y_rot_angle = torch.round(torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)).to(torch.long)
        neg_mask = (y_rot_angle < 0).long()
        mask = (y_rot_angle < -39).long()
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        return self.dynamic_lmk_faces_idx[y_rot_angle], self.dynamic_lmk_bary_coords[y_rot_angle]

    def forward(self, shape_params, expression_params, pose_params, eye_pose_params=None):
        batch_size = shape_params.shape[0]
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat(
            [pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params],
            dim=1,
        )
        template = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(
            betas,
            full_pose,
            template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights,
        )
        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(0).expand(batch_size, -1, -1)
        dyn_faces, dyn_bary = self._find_dynamic_lmk_idx_and_bcoords(full_pose)
        lmk_faces_idx = torch.cat([dyn_faces, lmk_faces_idx], dim=1)
        lmk_bary_coords = torch.cat([dyn_bary, lmk_bary_coords], dim=1)
        landmarks2d = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            self.full_lmk_faces_idx.unsqueeze(0).expand(batch_size, -1),
            self.full_lmk_bary_coords.unsqueeze(0).expand(batch_size, -1, -1),
        )
        return vertices, landmarks2d, landmarks3d
