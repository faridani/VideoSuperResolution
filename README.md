# Video Super Resolution

This project provides a PyTorch implementation of a state-of-the-art video super resolution model. A short sliding window of low resolution frames is processed to produce a high resolution output (e.g. 4K). The architecture combines modern techniques for spatial detail and temporal consistency.

## Key Features

- **Spatio‑Temporal Encoding** with 3D convolutions over a sequence of frames.
- **U‑Net Style Decoder** with pixel shuffle upsampling.
- **Transformer Bottleneck** for non‑local feature interactions.
- **Deformable Convolution** blocks for frame alignment.
- **Stable Diffusion Refinement** using a pretrained SDXL model.
- **CLIP‑Based Semantic Loss** in addition to perceptual and temporal losses.

## Training

The main training script is `video_sr.py`. It expects a directory of video files (e.g. MP4, AVI) rather than pre-extracted PNG frames.

```bash
python video_sr.py
```

Parameters such as window size, scaling factor and batch size can be adjusted at the bottom of the script.

## Requirements

- PyTorch
- diffusers
- OpenAI CLIP
- torchvision
- OpenCV

Install the required packages with:

```bash
pip install torch torchvision diffusers git+https://github.com/openai/CLIP.git opencv-python
```

## License

This project is released under the MIT License.
