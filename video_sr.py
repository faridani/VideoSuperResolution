import glob
import os
import random
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg16
import clip


VIDEO_EXTS: Tuple[str, ...] = (
    "*.mp4",
    "*.avi",
    "*.mov",
    "*.mkv",
    "*.mpg",
    "*.mpeg",
)


class VideoSuperResDataset(Dataset):
    """Dataset for video super-resolution training."""

    def __init__(
        self,
        video_dir: str,
        window_size: int = 5,
        scale: int = 4,
        transform: Optional[Callable] = None,
        degrade_level: float = 0.0,
    ) -> None:
        """Collect samples from all videos in ``video_dir``."""

        self.video_files: List[str] = [
            file
            for ext in VIDEO_EXTS
            for file in glob.glob(os.path.join(video_dir, ext))
        ]
        self.window_size = window_size
        self.scale = scale
        self.transform = transform
        self.degrade_level = degrade_level
        self.samples = self._collect_samples()

    def _collect_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for vfile in self.video_files:
            cap = cv2.VideoCapture(vfile)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            for start in range(0, frame_count - self.window_size + 1):
                samples.append((vfile, start))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _degrade(self, frame: np.ndarray) -> np.ndarray:
        """Apply random degradation to generate a low resolution frame."""

        if random.random() < 0.5 + self.degrade_level * 0.5:
            ksize = random.choice([3, 5])
            frame = cv2.GaussianBlur(frame, (ksize, ksize), 0)

        if random.random() < 0.5 + self.degrade_level * 0.5:
            noise_std = random.uniform(0, 10 + 20 * self.degrade_level)
            noise = np.random.normal(0, noise_std, frame.shape).astype(np.float32)
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < 0.3 * (1 + self.degrade_level):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(20, 50)]
            _, enc = cv2.imencode('.jpg', frame, encode_param)
            frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        if random.random() < 0.2 * (1 + self.degrade_level):
            tx = random.randint(-2, 2)
            ty = random.randint(-2, 2)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

        interp_methods = [
            cv2.INTER_AREA,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        interp = random.choice(interp_methods)
        lr = cv2.resize(
            frame,
            (frame.shape[1] // self.scale, frame.shape[0] // self.scale),
            interpolation=interp,
        )
        return lr

    def _read_window(self, video_file: str, start: int) -> np.ndarray:
        cap = cv2.VideoCapture(video_file)
        frames = []
        for i in range(start, start + self.window_size):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError(f"Could not read frame {i} from {video_file}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return np.stack(frames, axis=0)

    def _generate_lr_stack(self, hr_stack: np.ndarray) -> np.ndarray:
        lr_frames = [self._degrade(frame) for frame in hr_stack]
        return np.stack(lr_frames, axis=0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_file, start_idx = self.samples[idx]
        hr_frames = self._read_window(video_file, start_idx)
        lr_frames = self._generate_lr_stack(hr_frames)

        hr_frames = (hr_frames.astype(np.float32) / 255.0).transpose(3, 0, 1, 2)
        lr_frames = (lr_frames.astype(np.float32) / 255.0).transpose(3, 0, 1, 2)

        hr_tensor = torch.from_numpy(hr_frames)
        lr_tensor = torch.from_numpy(lr_frames)

        if self.transform:
            hr_tensor = self.transform(hr_tensor)
            lr_tensor = self.transform(lr_tensor)

        return lr_tensor, hr_tensor


class DeformableConv2d(nn.Module):
    """Deformable convolution using torchvision/mmcv if available."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        try:
            from torchvision.ops import DeformConv2d

            self.offset_conv = nn.Conv2d(
                in_channels,
                2 * kernel_size * kernel_size,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.dcn = DeformConv2d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            )
            self.use_dcn = True
        except Exception:
            try:
                from mmcv.ops import DeformConv2d  # type: ignore

                self.offset_conv = nn.Conv2d(
                    in_channels,
                    2 * kernel_size * kernel_size,
                    kernel_size=kernel_size,
                    padding=padding,
                )
                self.dcn = DeformConv2d(
                    in_channels, out_channels, kernel_size=kernel_size, padding=padding
                )
                self.use_dcn = True
            except Exception:
                # Fall back to standard convolution if neither library is available
                self.conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size, padding=padding
                )
                self.use_dcn = False

    def forward(self, x):
        if self.use_dcn:
            offset = self.offset_conv(x)
            return self.dcn(x, offset)
        return self.conv(x)


class Encoder3D(nn.Module):
    """Spatio-temporal encoder using 3D convolutions."""

    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        return out3, out2, out1


class TemporalFusion(nn.Module):
    """Collapse temporal dimension using attention based weighting."""

    def __init__(self, in_channels, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention
        self.fusion_conv = nn.Conv3d(
            in_channels, in_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)
        )
        if use_attention:
            self.attn_conv = nn.Conv3d(
                in_channels, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)
            )

    def forward(self, x):
        features = self.fusion_conv(x)
        if self.use_attention:
            attn = self.attn_conv(x)
            weights = torch.softmax(attn, dim=2)
            fused = torch.sum(features * weights, dim=2)
        else:
            fused = torch.mean(features, dim=2)
        return fused


class TransformerBottleneck(nn.Module):
    """Linformer style transformer encoder for efficiency."""

    def __init__(self, embed_dim, num_heads=4, proj_k=64):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.num_heads = num_heads
        self.proj_k = proj_k
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.E = nn.Parameter(torch.randn(proj_k, embed_dim))
        self.F = nn.Parameter(torch.randn(proj_k, embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # B L C
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm)
        K = self.k_proj(x_norm)
        V = self.v_proj(x_norm)
        K_lin = torch.einsum("blc,kc->blk", K, self.E)
        V_lin = torch.einsum("blc,kc->blk", V, self.F)
        attn_scores = torch.einsum("blc,blk->blk", Q, K_lin) / (C ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.einsum("blk,blk->blc", attn_weights, V_lin)
        out = self.out_proj(attn_out)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        out = x.permute(0, 2, 1).view(B, C, H, W)
        return out


class Decoder2D(nn.Module):
    """U-Net style decoder with pixel shuffle, deformable conv and skip fusion."""

    def __init__(self, in_channels, base_channels=64, scale=4):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.up2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(base_channels + base_channels, base_channels, kernel_size=3, padding=1)
        self.deform_conv = DeformableConv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)

    def forward(self, x, skip2, skip1):
        x = self.up1(x)
        x = torch.cat([x, skip2], dim=1)
        x = F.relu(self.conv1(x), inplace=True)
        x = self.up2(x)
        x = torch.cat([x, skip1], dim=1)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.deform_conv(x)
        out = self.final_conv(x)
        return out


class DiffusionRefinement(nn.Module):
    """Refine output using a pretrained SDXL UNet."""

    def __init__(self, device):
        super().__init__()
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float32
        )
        self.pipeline.to(device)
        for param in self.pipeline.unet.parameters():
            param.requires_grad = False
        self.timestep_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(3, 1, kernel_size=1),
            nn.Flatten(),
            nn.Sigmoid(),
        )
        self.max_timesteps = 50
        self.device = device

    def forward(self, x):
        B, C, H, W = x.shape
        t_weight = self.timestep_predictor(x).view(-1)
        timesteps = (t_weight * self.max_timesteps).long().clamp(1, self.max_timesteps)
        timesteps = timesteps.to(self.device)
        with torch.no_grad():
            noise_pred = self.pipeline.unet(x, timesteps=timesteps).sample
        refined = torch.clamp(x + noise_pred, 0, 1)
        return refined


class CLIPLoss(nn.Module):
    """Semantic loss using a pretrained CLIP image encoder."""

    def __init__(self, device):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def forward(self, pred, target):
        pred_resized = F.interpolate(pred, size=(224, 224), mode="bilinear", align_corners=False)
        target_resized = F.interpolate(target, size=(224, 224), mode="bilinear", align_corners=False)
        pred_norm = (pred_resized - self.mean) / self.std
        target_norm = (target_resized - self.mean) / self.std
        with torch.no_grad():
            pred_features = self.clip_model.encode_image(pred_norm)
            target_features = self.clip_model.encode_image(target_norm)
        loss = 1 - self.cosine_similarity(pred_features, target_features).mean()
        return loss


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss."""

    def __init__(self, device):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)
        loss = self.criterion(pred_features, target_features)
        return loss


def temporal_consistency_loss(pred_frames):
    loss = 0.0
    for t in range(pred_frames.size(1) - 1):
        loss += F.l1_loss(pred_frames[:, t + 1], pred_frames[:, t])
    return loss / (pred_frames.size(1) - 1)


def optical_flow_consistency_loss(pred_frames: torch.Tensor) -> torch.Tensor:
    """Penalize discrepancies after warping consecutive frames using optical flow."""
    loss = 0.0
    b, t, c, h, w = pred_frames.size()
    prev = pred_frames[:, 0]
    for i in range(1, t):
        next_frame = pred_frames[:, i]
        total_pair_loss = 0.0
        for b_idx in range(b):
            prev_np = (prev[b_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            next_np = (next_frame[b_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            prev_gray = cv2.cvtColor(prev_np, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(next_np, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            map_x = (grid_x + flow[..., 0]).astype(np.float32)
            map_y = (grid_y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(prev_np, map_x, map_y, cv2.INTER_LINEAR)
            warped_tensor = torch.from_numpy(warped.transpose(2, 0, 1) / 255.0).to(pred_frames.device)
            total_pair_loss += F.l1_loss(warped_tensor, next_frame[b_idx])
        loss += total_pair_loss / b
        prev = next_frame
    loss = loss / (t - 1)
    return loss


class VideoSuperResolutionDiffusionModel(nn.Module):
    """Full video super-resolution model integrating diffusion refinement."""

    def __init__(self, window_size=5, scale=4, base_channels=64, device="cuda"):
        super().__init__()
        self.window_size = window_size
        self.scale = scale
        self.encoder = Encoder3D(in_channels=3, base_channels=base_channels)
        self.temporal_fusion = TemporalFusion(in_channels=base_channels * 4)
        self.transformer_bottleneck = TransformerBottleneck(embed_dim=base_channels * 4)
        self.decoder = Decoder2D(in_channels=base_channels * 4, base_channels=base_channels, scale=scale)
        self.diffusion_refinement = DiffusionRefinement(device)

    def forward(self, x):
        feat3, feat2, feat1 = self.encoder(x)
        fused3 = self.temporal_fusion(feat3)
        fused2 = self.temporal_fusion(feat2)
        fused1 = self.temporal_fusion(feat1)
        bottleneck_features = self.transformer_bottleneck(fused3)
        sr = self.decoder(bottleneck_features, fused2, fused1)
        refined_sr = self.diffusion_refinement(sr)
        return refined_sr


def train_model(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    device: torch.device,
) -> None:
    """Simple training loop printing average loss per epoch."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    l1_loss_fn = nn.L1Loss()
    perceptual_loss_fn = PerceptualLoss(device)
    clip_loss_fn = CLIPLoss(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        # progressively increase degradation complexity
        if hasattr(dataloader.dataset, "degrade_level"):
            dataloader.dataset.degrade_level = min(1.0, epoch / num_epochs)
        for lr_seq, hr_seq in dataloader:
            lr_seq = lr_seq.to(device)
            hr_seq = hr_seq.to(device)

            optimizer.zero_grad()
            sr = model(lr_seq)
            hr_central = hr_seq[:, :, hr_seq.size(2) // 2, :, :]

            loss_l1 = l1_loss_fn(sr, hr_central)
            loss_perc = perceptual_loss_fn(sr, hr_central)
            loss_clip = clip_loss_fn(sr, hr_central)

            sr_seq = sr.unsqueeze(1).repeat(1, lr_seq.size(2), 1, 1, 1)
            loss_temp = temporal_consistency_loss(sr_seq)
            loss_flow = optical_flow_consistency_loss(sr_seq.detach())

            total_loss = (
                loss_l1
                + 0.1 * loss_perc
                + 0.1 * loss_clip
                + 0.5 * loss_temp
                + 0.1 * loss_flow
            )
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = F.mse_loss(pred, target).item()
    return 20 * np.log10(1.0 / np.sqrt(mse + 1e-8))


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> None:
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for lr_seq, hr_seq in dataloader:
            lr_seq = lr_seq.to(device)
            hr_seq = hr_seq.to(device)
            sr = model(lr_seq)
            hr_central = hr_seq[:, :, hr_seq.size(2) // 2, :, :]
            total_psnr += psnr(sr, hr_central)
    avg_psnr = total_psnr / len(dataloader)
    print(f"Validation PSNR: {avg_psnr:.2f}dB")


if __name__ == "__main__":
    video_data_dir = "./videos"
    window_size = 5
    scale = 4
    num_epochs = 50
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = VideoSuperResDataset(video_data_dir, window_size=window_size, scale=scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = VideoSuperResolutionDiffusionModel(window_size=window_size, scale=scale, base_channels=64, device=device)
    train_model(model, dataloader, num_epochs, device)
    evaluate_model(model, dataloader, device)

    try:
        import optuna

        def objective(trial):
            lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            train_model(model, dataloader, 1, device)
            return 0.0

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)
    except Exception:
        print("Optuna not available; skipping hyperparameter tuning")

