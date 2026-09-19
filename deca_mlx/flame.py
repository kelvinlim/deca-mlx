"""FLAME decoder ported from official DECA ``decalib/models/FLAME.py``."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import DECAConfig
from .flame_io import flame_to_numpy as _to_np
from .flame_io import load_flame_pickle as _load_flame_pickle
from .lbs import batch_rodrigues, lbs, rot_mat_to_euler, vertices2landmarks


class FLAME(nn.Module):
    def __init__(self, config: DECAConfig):
        super().__init__()
        if not config.flame_model_path.exists():
            raise FileNotFoundError(
                f"FLAME model not found at {config.flame_model_path}. "
                "Register at https://flame.is.tue.mpg.de/ and place generic_model.pkl in data/."
            )
        flame_model = SimpleNamespace(**_load_flame_pickle(config.flame_model_path))

        faces = _to_np(flame_model.f, dtype=np.int32)
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

        self.faces_tensor = mx.array(faces)
        self.v_template = mx.array(v_template)
        self.shapedirs = mx.array(shapedirs)
        self.posedirs = mx.array(posedirs)
        self.J_regressor = mx.array(j_regressor)
        self.parents = mx.array(parents)
        self.lbs_weights = mx.array(lbs_weights)
        self.eye_pose = mx.zeros((1, 6))
        self.neck_pose = mx.zeros((1, 3))

        lmk = np.load(config.flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")
        lmk = lmk[()]
        self.lmk_faces_idx = mx.array(np.array(lmk["static_lmk_faces_idx"], dtype=np.int32))
        self.lmk_bary_coords = mx.array(np.array(lmk["static_lmk_bary_coords"], dtype=np.float32))
        self.dynamic_lmk_faces_idx = mx.array(np.array(lmk["dynamic_lmk_faces_idx"], dtype=np.int32))
        self.dynamic_lmk_bary_coords = mx.array(np.array(lmk["dynamic_lmk_bary_coords"], dtype=np.float32))
        self.full_lmk_faces_idx = mx.array(np.array(lmk["full_lmk_faces_idx"], dtype=np.int32))
        self.full_lmk_bary_coords = mx.array(np.array(lmk["full_lmk_bary_coords"], dtype=np.float32))

        neck_chain = []
        curr = 1
        while curr != -1:
            neck_chain.append(curr)
            curr = int(parents[curr])
        self.neck_kin_chain = mx.array(np.array(neck_chain, dtype=np.int32))

    def _find_dynamic_lmk_idx_and_bcoords(self, pose: mx.array) -> tuple[mx.array, mx.array]:
        batch_size = pose.shape[0]
        aa_pose = mx.take(pose.reshape((batch_size, -1, 3)), self.neck_kin_chain, axis=1)
        rot_mats = batch_rodrigues(aa_pose.reshape((-1, 3))).reshape((batch_size, -1, 3, 3))
        rel_rot_mat = mx.broadcast_to(mx.eye(3)[None, :, :], (batch_size, 3, 3))
        for idx in range(self.neck_kin_chain.shape[0]):
            rel_rot_mat = rot_mats[:, idx] @ rel_rot_mat

        # Match FLAME.py (no extra negation used by the unused lbs.py helper).
        y_rot_angle = mx.round(mx.minimum(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, 39.0))
        y_rot_angle = y_rot_angle.astype(mx.int32)
        neg_mask = (y_rot_angle < 0).astype(mx.int32)
        mask = (y_rot_angle < -39).astype(mx.int32)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = neg_mask * neg_vals + (1 - neg_mask) * y_rot_angle
        dyn_faces = mx.take(self.dynamic_lmk_faces_idx, y_rot_angle, axis=0)
        dyn_b = mx.take(self.dynamic_lmk_bary_coords, y_rot_angle, axis=0)
        return dyn_faces, dyn_b

    def seletec_3d68(self, vertices: mx.array) -> mx.array:
        bz = vertices.shape[0]
        return vertices2landmarks(
            vertices,
            self.faces_tensor,
            mx.broadcast_to(self.full_lmk_faces_idx, (bz, self.full_lmk_faces_idx.shape[-1])),
            mx.broadcast_to(self.full_lmk_bary_coords, (bz,) + self.full_lmk_bary_coords.shape[-2:]),
        )

    def __call__(
        self,
        shape_params: mx.array,
        expression_params: mx.array,
        pose_params: mx.array,
        eye_pose_params: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        batch_size = shape_params.shape[0]
        if eye_pose_params is None:
            eye_pose_params = mx.broadcast_to(self.eye_pose, (batch_size, 6))
        betas = mx.concatenate([shape_params, expression_params], axis=1)
        neck = mx.broadcast_to(self.neck_pose, (batch_size, 3))
        full_pose = mx.concatenate([pose_params[:, :3], neck, pose_params[:, 3:], eye_pose_params], axis=1)
        template = mx.broadcast_to(self.v_template[None, ...], (batch_size,) + self.v_template.shape)
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

        lmk_faces_idx = mx.broadcast_to(self.lmk_faces_idx[None, :], (batch_size, self.lmk_faces_idx.shape[0]))
        lmk_bary_coords = mx.broadcast_to(self.lmk_bary_coords[None, ...], (batch_size,) + self.lmk_bary_coords.shape)
        dyn_faces, dyn_bary = self._find_dynamic_lmk_idx_and_bcoords(full_pose)
        lmk_faces_idx = mx.concatenate([dyn_faces, lmk_faces_idx], axis=1)
        lmk_bary_coords = mx.concatenate([dyn_bary, lmk_bary_coords], axis=1)
        landmarks2d = vertices2landmarks(vertices, self.faces_tensor, lmk_faces_idx, lmk_bary_coords)
        landmarks3d = vertices2landmarks(
            vertices,
            self.faces_tensor,
            mx.broadcast_to(self.full_lmk_faces_idx, (batch_size, self.full_lmk_faces_idx.shape[-1])),
            mx.broadcast_to(self.full_lmk_bary_coords, (batch_size,) + self.full_lmk_bary_coords.shape[-2:]),
        )
        return vertices, landmarks2d, landmarks3d
