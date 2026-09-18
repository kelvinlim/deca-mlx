# Model assets

Already vendored from the official DECA repo:

- `head_template.obj`
- `landmark_embedding.npy`
- `fixed_displacement_256.npy`
- `texture_data_256.npy`
- `uv_face_mask.png`
- `uv_face_eye_mask.png`
- `mean_texture.jpg`

Optional (gitignored), used only for the 6th vis panel:

- `FLAME_albedo_from_BFM.npz` — official DECA albedo PCA. Without it the lit-albedo panel uses `mean_texture.jpg`.

Download locally (gitignored):

- `deca_model.tar` — official DECA weights, Google Drive id `1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje`
- `deca_mlx.safetensors` — created by `python -m deca_mlx.convert`
- `generic_model.pkl` — FLAME 2020, from https://flame.is.tue.mpg.de/
