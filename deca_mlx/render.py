"""Software rasterizer matching official DECA ``SRenderY`` visualization."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .config import DECAConfig, default_config
from .export import load_obj, vertex_normals

SHAPE_COLOR = np.array([180.0, 180.0, 180.0], dtype=np.float32) / 255.0
SHAPE_LIGHT_DIRS = np.array(
    [
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
SHAPE_LIGHT_INTENSITY = 1.7
LANDMARK_ENDS = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1
SH_FACTOR = np.array(
    [
        1.0 / np.sqrt(4.0 * np.pi),
        ((2.0 * np.pi) / 3.0) * np.sqrt(3.0 / (4.0 * np.pi)),
        ((2.0 * np.pi) / 3.0) * np.sqrt(3.0 / (4.0 * np.pi)),
        ((2.0 * np.pi) / 3.0) * np.sqrt(3.0 / (4.0 * np.pi)),
        (np.pi / 4.0) * 3.0 * np.sqrt(5.0 / (12.0 * np.pi)),
        (np.pi / 4.0) * 3.0 * np.sqrt(5.0 / (12.0 * np.pi)),
        (np.pi / 4.0) * 3.0 * np.sqrt(5.0 / (12.0 * np.pi)),
        (np.pi / 4.0) * (3.0 / 2.0) * np.sqrt(5.0 / (12.0 * np.pi)),
        (np.pi / 4.0) * (1.0 / 2.0) * np.sqrt(5.0 / (4.0 * np.pi)),
    ],
    dtype=np.float32,
)


def orth_proj(verts: np.ndarray, cam: np.ndarray) -> np.ndarray:
    cam = cam.reshape(3)
    out = verts.copy()
    out[:, :2] = out[:, :2] + cam[1:]
    out = out * cam[0]
    out[:, 1:] = -out[:, 1:]
    return out


def generate_triangles(height: int, width: int, margin_x: int = 2, margin_y: int = 5) -> np.ndarray:
    triangles = []
    for x in range(margin_x, width - 1 - margin_x):
        for y in range(margin_y, height - 1 - margin_y):
            triangles.append([y * width + x, y * width + x + 1, (y + 1) * width + x])
            triangles.append([y * width + x + 1, (y + 1) * width + x + 1, (y + 1) * width + x])
    triangles = np.asarray(triangles, dtype=np.int32)
    return triangles[:, [0, 2, 1]]


def face_vertices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return vertices[faces]


def barycentric_interp(values: np.ndarray, faces: np.ndarray, face_idx: np.ndarray, bary: np.ndarray) -> np.ndarray:
    corners = values[faces[face_idx]]
    return np.sum(corners * bary[..., None], axis=1)


def add_directionlight(normals: np.ndarray, light_dirs: np.ndarray | None = None, intensity: float = SHAPE_LIGHT_INTENSITY) -> np.ndarray:
    dirs = SHAPE_LIGHT_DIRS if light_dirs is None else light_dirs
    dirs = dirs / np.clip(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-8, None)
    ndotl = np.clip(normals @ dirs.T, 0.0, 1.0)
    return (ndotl.mean(axis=1) * intensity).astype(np.float32)


def add_shlight(normals: np.ndarray, sh_coeff: np.ndarray) -> np.ndarray:
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    basis = np.stack(
        [
            np.ones_like(nx),
            nx,
            ny,
            nz,
            nx * ny,
            nx * nz,
            ny * nz,
            nx * nx - ny * ny,
            3.0 * nz * nz - 1.0,
        ],
        axis=-1,
    )
    basis = basis * SH_FACTOR
    sh = sh_coeff.reshape(9, 3)
    return basis @ sh


def grid_sample(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Bilinear sample ``image`` [H, W, C] with PyTorch ``align_corners=False`` grid [H, W, 2]."""
    height, width = image.shape[:2]
    xs = (grid[..., 0] + 1.0) * width / 2.0 - 0.5
    ys = (grid[..., 1] + 1.0) * height / 2.0 - 0.5
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xs - x0
    wy = ys - y0
    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)
    i00 = image[y0, x0]
    i01 = image[y0, x1]
    i10 = image[y1, x0]
    i11 = image[y1, x1]
    top = i00 * (1.0 - wx[..., None]) + i01 * wx[..., None]
    bot = i10 * (1.0 - wx[..., None]) + i11 * wx[..., None]
    return top * (1.0 - wy[..., None]) + bot * wy[..., None]


def rasterize(
    vertices: np.ndarray,
    faces: np.ndarray,
    attributes: np.ndarray,
    height: int,
    width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Z-buffer rasterize NDC vertices ``[V, 3]`` and per-face attributes ``[F, 3, D]``."""
    width = height if width is None else width
    pixels = np.empty((vertices.shape[0], 2), dtype=np.float32)
    pixels[:, 0] = vertices[:, 0] * (width / 2.0) + (width / 2.0)
    pixels[:, 1] = vertices[:, 1] * (height / 2.0) + (height / 2.0)
    depth = vertices[:, 2].astype(np.float32)

    dim = attributes.shape[-1]
    image = np.zeros((height, width, dim), dtype=np.float32)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)

    v0, v1, v2 = faces[:, 0], faces[:, 1], faces[:, 2]
    p0, p1, p2 = pixels[v0], pixels[v1], pixels[v2]
    d0, d1, d2 = depth[v0], depth[v1], depth[v2]
    a0, a1, a2 = attributes[:, 0], attributes[:, 1], attributes[:, 2]

    min_x = np.clip(np.floor(np.minimum(np.minimum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32), 0, width - 1)
    max_x = np.clip(np.ceil(np.maximum(np.maximum(p0[:, 0], p1[:, 0]), p2[:, 0])).astype(np.int32), 0, width - 1)
    min_y = np.clip(np.floor(np.minimum(np.minimum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32), 0, height - 1)
    max_y = np.clip(np.ceil(np.maximum(np.maximum(p0[:, 1], p1[:, 1]), p2[:, 1])).astype(np.int32), 0, height - 1)

    for i in range(faces.shape[0]):
        x0, x1 = int(min_x[i]), int(max_x[i])
        y0, y1 = int(min_y[i]), int(max_y[i])
        if x1 <= x0 or y1 <= y0:
            continue
        a, b, c = p0[i], p1[i], p2[i]
        denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denom) < 1e-8:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        grid_x, grid_y = np.meshgrid(xs, ys)
        w0 = ((b[1] - c[1]) * (grid_x - c[0]) + (c[0] - b[0]) * (grid_y - c[1])) / denom
        w1 = ((c[1] - a[1]) * (grid_x - c[0]) + (a[0] - c[0]) * (grid_y - c[1])) / denom
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(mask):
            continue
        z = w0 * d0[i] + w1 * d1[i] + w2 * d2[i]
        yy = grid_y[mask]
        xx = grid_x[mask]
        zz = z[mask]
        closer = zz < zbuf[yy, xx]
        if not np.any(closer):
            continue
        ww0 = w0[mask][closer]
        ww1 = w1[mask][closer]
        ww2 = w2[mask][closer]
        yy, xx, zz = yy[closer], xx[closer], zz[closer]
        zbuf[yy, xx] = zz
        image[yy, xx] = ww0[:, None] * a0[i] + ww1[:, None] * a1[i] + ww2[:, None] * a2[i]

    alpha = np.isfinite(zbuf).astype(np.float32)
    return image, alpha


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def plot_kpts(image: np.ndarray, kpts: np.ndarray) -> np.ndarray:
    canvas = image.copy()
    radius = max(int(min(canvas.shape[0], canvas.shape[1]) / 200), 1)
    for i in range(kpts.shape[0]):
        start = kpts[i, :2]
        if kpts.shape[1] >= 4:
            color = (0, 255, 0) if kpts[i, 3] > 0.5 else (255, 0, 0)
        else:
            color = (0, 255, 0)
        if i in LANDMARK_ENDS:
            continue
        end = kpts[i + 1, :2]
        cv2.line(
            canvas,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            (255, 255, 255),
            radius,
        )
        cv2.circle(canvas, (int(start[0]), int(start[1])), radius, color, radius * 2)
    return canvas


def vis_landmarks(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    canvas = _to_uint8(image)
    pts = landmarks.astype(np.float32).copy()
    pts[:, 0] = pts[:, 0] * canvas.shape[1] / 2.0 + canvas.shape[1] / 2.0
    pts[:, 1] = pts[:, 1] * canvas.shape[0] / 2.0 + canvas.shape[0] / 2.0
    return plot_kpts(canvas, pts)


def hstack_panels(panels: list[np.ndarray], size: int = 224) -> np.ndarray:
    resized = []
    for panel in panels:
        if panel.shape[0] != size or panel.shape[1] != size:
            panel = cv2.resize(panel, (size, size), interpolation=cv2.INTER_LINEAR)
        resized.append(panel)
    return np.concatenate(resized, axis=1)


class Renderer:
    def __init__(self, config: DECAConfig | None = None):
        cfg = config or default_config()
        verts, uvcoords, faces, uv_faces = load_obj(cfg.topology_path)
        self.faces = faces.astype(np.int32)
        self.uv_faces = uv_faces.astype(np.int32)
        self.template_verts = verts.astype(np.float32)
        uvs = np.concatenate([uvcoords, np.ones((uvcoords.shape[0], 1), dtype=np.float32)], axis=1)
        uvs = uvs * 2.0 - 1.0
        uvs[:, 1] = -uvs[:, 1]
        self.uvcoords = uvs
        self.uv_size = cfg.uv_size
        self.image_size = cfg.image_size
        self.dense_faces = generate_triangles(cfg.uv_size, cfg.uv_size)
        self.fixed_uv_dis = None
        if cfg.fixed_displacement_path.exists():
            self.fixed_uv_dis = np.load(cfg.fixed_displacement_path).astype(np.float32)
        self.uv_face_eye_mask = None
        if cfg.face_eye_mask_path.exists():
            mask = np.array(Image.open(cfg.face_eye_mask_path).convert("L"), dtype=np.float32) / 255.0
            mask = cv2.resize(mask, (cfg.uv_size, cfg.uv_size), interpolation=cv2.INTER_LINEAR)
            self.uv_face_eye_mask = mask
        self.mean_albedo = None
        if cfg.mean_tex_path.exists():
            albedo = np.array(Image.open(cfg.mean_tex_path).convert("RGB"), dtype=np.float32) / 255.0
            self.mean_albedo = cv2.resize(albedo, (cfg.uv_size, cfg.uv_size), interpolation=cv2.INTER_LINEAR)
        self.tex_mean = None
        self.tex_basis = None
        if cfg.tex_path.exists():
            space = np.load(cfg.tex_path)
            self.tex_mean = space["MU"].reshape(-1).astype(np.float32)
            self.tex_basis = space["PC"].reshape(-1, 199)[:, : cfg.n_tex].astype(np.float32)
        self.full_lmk_faces_idx = None
        self.full_lmk_bary_coords = None
        if cfg.flame_lmk_embedding_path.exists():
            lmk = np.load(cfg.flame_lmk_embedding_path, allow_pickle=True, encoding="latin1")[()]
            self.full_lmk_faces_idx = np.array(lmk["full_lmk_faces_idx"], dtype=np.int32).reshape(-1)
            self.full_lmk_bary_coords = np.array(lmk["full_lmk_bary_coords"], dtype=np.float32)
            if self.full_lmk_bary_coords.ndim == 3:
                self.full_lmk_bary_coords = self.full_lmk_bary_coords[0]

    def decode_albedo(self, tex_code: np.ndarray | None) -> np.ndarray:
        if self.tex_mean is not None and tex_code is not None:
            texture = self.tex_mean + self.tex_basis @ tex_code.reshape(-1)
            texture = texture.reshape(512, 512, 3)[..., ::-1]
            return cv2.resize(np.clip(texture, 0.0, 1.0), (self.uv_size, self.uv_size), interpolation=cv2.INTER_LINEAR)
        if self.mean_albedo is not None:
            return self.mean_albedo
        return np.full((self.uv_size, self.uv_size, 3), 0.7, dtype=np.float32)

    def world2uv(self, values: np.ndarray) -> np.ndarray:
        attrs = face_vertices(values, self.faces)
        image, _ = rasterize(self.uvcoords, self.uv_faces, attrs, self.uv_size, self.uv_size)
        return image

    def displacement2normal(self, uv_z: np.ndarray, verts: np.ndarray, normals: np.ndarray) -> np.ndarray:
        uv_verts = self.world2uv(verts)
        uv_normals = self.world2uv(normals)
        uv_normals = uv_normals / np.clip(np.linalg.norm(uv_normals, axis=-1, keepdims=True), 1e-8, None)
        if uv_z.ndim == 3:
            uv_z = uv_z[..., 0] if uv_z.shape[-1] == 1 else uv_z[0]
        if self.uv_face_eye_mask is not None:
            uv_z = uv_z * self.uv_face_eye_mask
        offset = uv_z
        if self.fixed_uv_dis is not None:
            offset = offset + self.fixed_uv_dis
        detail_verts = uv_verts + offset[..., None] * uv_normals
        detail_normals = vertex_normals(detail_verts.reshape(-1, 3), self.dense_faces)
        detail_normals = detail_normals.reshape(self.uv_size, self.uv_size, 3)
        if self.uv_face_eye_mask is not None:
            mask = self.uv_face_eye_mask[..., None]
            detail_normals = detail_normals * mask + uv_normals * (1.0 - mask)
        return detail_normals / np.clip(np.linalg.norm(detail_normals, axis=-1, keepdims=True), 1e-8, None)

    def _shade_shape(self, albedo: np.ndarray, world_normals: np.ndarray, trans_normals: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        shade = add_directionlight(world_normals.reshape(-1, 3)).reshape(albedo.shape[:2])
        pos_mask = (trans_normals[..., 2] < 0.15).astype(np.float32)
        coverage = alpha * pos_mask
        image = albedo * shade[..., None]
        return np.clip(image * coverage[..., None], 0.0, 1.0)

    def raster_mesh(self, verts: np.ndarray, trans_verts: np.ndarray, size: int | None = None) -> dict[str, np.ndarray]:
        size = self.image_size if size is None else size
        shifted = trans_verts.copy()
        shifted[:, 2] = shifted[:, 2] + 10.0
        world_n = vertex_normals(verts, self.faces)
        trans_n = vertex_normals(shifted, self.faces)
        colors = np.broadcast_to(SHAPE_COLOR, verts.shape).copy()
        attrs = np.concatenate(
            [
                face_vertices(colors, self.faces),
                face_vertices(trans_n, self.faces),
                face_vertices(verts, self.faces),
                face_vertices(world_n, self.faces),
                face_vertices(self.uvcoords, self.uv_faces),
            ],
            axis=-1,
        )
        raster, alpha = rasterize(shifted, self.faces, attrs, size, size)
        return {
            "albedo": raster[..., 0:3],
            "trans_normals": raster[..., 3:6],
            "world_verts": raster[..., 6:9],
            "world_normals": raster[..., 9:12],
            "uv": raster[..., 12:15],
            "alpha": alpha,
            "vertex_normals": world_n,
            "transformed_normals": trans_n,
        }

    def render_shape(
        self,
        verts: np.ndarray,
        trans_verts: np.ndarray | None = None,
        cam: np.ndarray | None = None,
        detail_normals: np.ndarray | None = None,
        size: int | None = None,
    ) -> np.ndarray:
        if trans_verts is None:
            if cam is None:
                raise ValueError("render_shape needs trans_verts or cam")
            trans_verts = orth_proj(verts, cam)
        buffers = self.raster_mesh(verts, trans_verts, size=size)
        world_n = buffers["world_normals"]
        if detail_normals is not None:
            sampled = grid_sample(detail_normals, buffers["uv"][..., :2])
            world_n = sampled * buffers["alpha"][..., None] + world_n * (1.0 - buffers["alpha"][..., None])
        image = self._shade_shape(buffers["albedo"], world_n, buffers["trans_normals"], buffers["alpha"])
        return _to_uint8(image)

    def render_albedo(
        self,
        verts: np.ndarray,
        trans_verts: np.ndarray,
        albedo: np.ndarray,
        lights: np.ndarray,
        size: int | None = None,
    ) -> np.ndarray:
        buffers = self.raster_mesh(verts, trans_verts, size=size)
        sampled = grid_sample(albedo, buffers["uv"][..., :2])
        shading = add_shlight(buffers["world_normals"], lights)
        image = sampled * shading
        image = image * buffers["alpha"][..., None]
        return _to_uint8(image)

    def visualize(
        self,
        image: np.ndarray,
        verts: np.ndarray,
        trans_verts: np.ndarray,
        landmarks2d: np.ndarray,
        landmarks3d: np.ndarray,
        lights: np.ndarray | None = None,
        tex_code: np.ndarray | None = None,
        displacement: np.ndarray | None = None,
        size: int = 224,
    ) -> np.ndarray:
        buffers = self.raster_mesh(verts, trans_verts, size=size)
        coarse = self._shade_shape(buffers["albedo"], buffers["world_normals"], buffers["trans_normals"], buffers["alpha"])
        panels = [
            _to_uint8(image if image.shape[-1] == 3 else np.transpose(image, (1, 2, 0))),
            vis_landmarks(image, landmarks2d),
        ]
        lmk3d = landmarks3d
        if lmk3d.shape[-1] == 3:
            if self.full_lmk_faces_idx is not None:
                n68 = barycentric_interp(
                    buffers["transformed_normals"],
                    self.faces,
                    self.full_lmk_faces_idx,
                    self.full_lmk_bary_coords,
                )
                vis = (n68[:, 2] < 0.1).astype(np.float32)
            else:
                px = np.clip((lmk3d[:, 0] * size / 2.0 + size / 2.0).astype(np.int32), 0, size - 1)
                py = np.clip((lmk3d[:, 1] * size / 2.0 + size / 2.0).astype(np.int32), 0, size - 1)
                vis = (buffers["trans_normals"][py, px, 2] < 0.1).astype(np.float32)
            lmk3d = np.concatenate([lmk3d, vis[:, None]], axis=1)
        panels.append(vis_landmarks(image, lmk3d))
        panels.append(_to_uint8(coarse))

        if displacement is not None:
            uv_z = displacement[0] if displacement.ndim == 3 else displacement
            detail_n = self.displacement2normal(uv_z, verts, buffers["vertex_normals"])
            sampled = grid_sample(detail_n, buffers["uv"][..., :2])
            sampled = sampled * buffers["alpha"][..., None] + buffers["world_normals"] * (1.0 - buffers["alpha"][..., None])
            detail = self._shade_shape(buffers["albedo"], sampled, buffers["trans_normals"], buffers["alpha"])
            panels.append(_to_uint8(detail))
        else:
            panels.append(_to_uint8(coarse))

        albedo = self.decode_albedo(tex_code)
        if lights is None:
            lights = np.zeros((9, 3), dtype=np.float32)
            lights[0] = 1.0
        sampled = grid_sample(albedo, buffers["uv"][..., :2])
        shading = add_shlight(buffers["world_normals"], lights)
        rendered = np.clip(sampled * shading * buffers["alpha"][..., None], 0.0, 1.0)
        panels.append(_to_uint8(rendered))
        return hstack_panels(panels, size=size)


def render_shape(verts: np.ndarray, faces: np.ndarray, cam: np.ndarray, size: int = 224) -> np.ndarray:
    """Gray shaded preview used by scripts; prefers the topology renderer when faces match."""
    renderer = Renderer()
    trans = orth_proj(verts, cam)
    if faces is not None and faces.shape == renderer.faces.shape:
        return renderer.render_shape(verts, trans, size=size)
    normals = vertex_normals(verts, faces)
    shade = add_directionlight(normals)
    colors = np.broadcast_to(SHAPE_COLOR, verts.shape) * shade[:, None]
    attrs = face_vertices(colors, faces)
    trans = trans.copy()
    trans[:, 2] = trans[:, 2] + 10.0
    image, alpha = rasterize(trans, faces, attrs, size, size)
    return _to_uint8(image * alpha[..., None])


def visualize_reconstruction(
    image: np.ndarray,
    verts: np.ndarray,
    trans_verts: np.ndarray,
    landmarks2d: np.ndarray,
    landmarks3d: np.ndarray,
    lights: np.ndarray | None = None,
    tex_code: np.ndarray | None = None,
    displacement: np.ndarray | None = None,
    renderer: Renderer | None = None,
    size: int = 224,
) -> np.ndarray:
    renderer = renderer or Renderer()
    return renderer.visualize(
        image,
        verts,
        trans_verts,
        landmarks2d,
        landmarks3d,
        lights=lights,
        tex_code=tex_code,
        displacement=displacement,
        size=size,
    )
