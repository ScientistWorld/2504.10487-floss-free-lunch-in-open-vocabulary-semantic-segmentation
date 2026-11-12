"""Frozen NACLIP dense features and CLIP text embeddings for ADE20K.

This is the fixed open-vocabulary segmentation backbone of the gym. It follows
the NACLIP settings used by the FLOSS paper for ADE20K (OpenAI CLIP ViT-B/16,
no PAMR refinement): the last transformer block uses key-key attention plus a
Gaussian neighbourhood bias (std 5) and drops its residual and FFN; images are
resized to a short side of 336 and processed with 224x224 sliding windows at
stride 112.

To make every window share one patch grid, the long side is rounded to the
nearest 224 + 112*j (aspect change at most ~14%, under 7% for typical 4:3 images). The per-patch output is the
average, over overlapping windows, of the L2-normalised patch embeddings, so
`features @ text_embedding` equals the window-averaged cosine logits of the
original pipeline before upsampling.

Outputs per image split (flat, float16):
  <prefix>_feats.npy   [total_patches, 512]
  <prefix>_index.npz   ids, hw [N,2] (patch grid), offsets [N+1],
                       label_hw [N,2] (= 8*hw), label_offsets [N+1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ADE20K_CLASSES = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest of drawers", "counter", "sand", "sink", "skyscraper",
    "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table",
    "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table",
    "toilet", "flower", "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island",
    "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus", "towel", "light",
    "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman",
    "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy",
    "washer", "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag",
    "minibike", "cradle", "oven", "ball", "food", "step", "tank", "trade name", "microwave", "pot",
    "animal", "bicycle", "lake", "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic light", "tray", "ashcan", "fan", "pier", "crt screen", "plate", "monitor",
    "bulletin board", "shower", "radiator", "glass", "clock", "flag",
]

# CLIP's 80 ImageNet templates, in the order of the paper's Appendix A.7.
IMAGENET_TEMPLATES = [
    "a bad photo of a {}.", "a photo of many {}.", "a sculpture of a {}.",
    "a photo of the hard to see {}.", "a low resolution photo of the {}.", "a rendering of a {}.",
    "graffiti of a {}.", "a bad photo of the {}.", "a cropped photo of the {}.", "a tattoo of a {}.",
    "the embroidered {}.", "a photo of a hard to see {}.", "a bright photo of a {}.",
    "a photo of a clean {}.", "a photo of a dirty {}.", "a dark photo of the {}.",
    "a drawing of a {}.", "a photo of my {}.", "the plastic {}.", "a photo of the cool {}.",
    "a close-up photo of a {}.", "a black and white photo of the {}.", "a painting of the {}.",
    "a painting of a {}.", "a pixelated photo of the {}.", "a sculpture of the {}.",
    "a bright photo of the {}.", "a cropped photo of a {}.", "a plastic {}.",
    "a photo of the dirty {}.", "a jpeg corrupted photo of a {}.", "a blurry photo of the {}.",
    "a photo of the {}.", "a good photo of the {}.", "a rendering of the {}.",
    "a {} in a video game.", "a photo of one {}.", "a doodle of a {}.",
    "a close-up photo of the {}.", "a photo of a {}.", "the origami {}.",
    "the {} in a video game.", "a sketch of a {}.", "a doodle of the {}.", "a origami {}.",
    "a low resolution photo of a {}.", "the toy {}.", "a rendition of the {}.",
    "a photo of the clean {}.", "a photo of a large {}.", "a rendition of a {}.",
    "a photo of a nice {}.", "a photo of a weird {}.", "a blurry photo of a {}.",
    "a cartoon {}.", "art of a {}.", "a sketch of the {}.", "a embroidered {}.",
    "a pixelated photo of a {}.", "itap of the {}.", "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.", "a plushie {}.", "a photo of the nice {}.",
    "a photo of the small {}.", "a photo of the weird {}.", "the cartoon {}.", "art of the {}.",
    "a drawing of the {}.", "a photo of the large {}.", "a black and white photo of a {}.",
    "the plushie {}.", "a dark photo of a {}.", "itap of a {}.", "graffiti of the {}.",
    "a toy {}.", "itap of my {}.", "a photo of a cool {}.", "a photo of a small {}.",
    "a tattoo of the {}.",
]
assert len(ADE20K_CLASSES) == 150 and len(IMAGENET_TEMPLATES) == 80

CLIP_MEAN = np.array([122.771, 116.746, 104.094], dtype=np.float32)
CLIP_STD = np.array([68.501, 66.632, 70.323], dtype=np.float32)
SHORT_SIDE = 336
CROP = 224
STRIDE = 112
PATCH = 16
LABEL_SCALE = 8  # label pixels per patch side (half of the 336-short-side resolution)
GAUSSIAN_STD = 5.0


def resized_hw(width: int, height: int) -> tuple[int, int]:
    """(H, W) of the resized image: short side 336, long side 224 + 112*j."""
    scale = SHORT_SIDE / min(width, height)
    long_side = max(width, height) * scale
    j = max(1, int(round((long_side - CROP) / STRIDE)))
    long_r = CROP + STRIDE * j
    return (SHORT_SIDE, long_r) if width >= height else (long_r, SHORT_SIDE)


class NaclipEncoder:
    def __init__(self, model_dir: str | Path, device: torch.device):
        from transformers import CLIPModel, CLIPTokenizer

        self.device = device
        self.model = CLIPModel.from_pretrained(str(model_dir), torch_dtype=torch.float32).to(device).eval()
        self.tokenizer = CLIPTokenizer.from_pretrained(str(model_dir))
        n = CROP // PATCH
        ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
        pos = torch.stack([ys.flatten(), xs.flatten()], dim=1).float()
        dist2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
        omega = torch.zeros(n * n + 1, n * n + 1)
        omega[1:, 1:] = torch.exp(-dist2 / (2.0 * GAUSSIAN_STD**2))
        self.omega = omega.to(device)

    @torch.no_grad()
    def encode_text(self, class_names: list[str], templates: list[str]) -> np.ndarray:
        """[M, K, 512] L2-normalised single-template class embeddings."""
        out = []
        for template in templates:
            prompts = [template.format(c) for c in class_names]
            tok = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
            emb = self.model.get_text_features(**tok).float()
            out.append(F.normalize(emb, dim=-1))
        return torch.stack(out).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def crop_features(self, pixels: torch.Tensor) -> torch.Tensor:
        """[B,3,224,224] normalised crops -> [B,14,14,512] L2-normalised NACLIP patch features."""
        vis = self.model.vision_model
        x = vis.embeddings(pixels)
        x = vis.pre_layrnorm(x)
        for layer in vis.encoder.layers[:-1]:
            out = layer(x, None, None)
            x = out[0] if isinstance(out, tuple) else out
        last = vis.encoder.layers[-1]
        h = last.layer_norm1(x)
        attn = last.self_attn
        b, n_tok, c = h.shape
        heads = attn.num_heads
        hd = c // heads
        k = attn.k_proj(h).view(b, n_tok, heads, hd).transpose(1, 2)
        v = attn.v_proj(h).view(b, n_tok, heads, hd).transpose(1, 2)
        weights = (k @ k.transpose(-1, -2)) * (hd**-0.5) + self.omega
        weights = weights.softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b, n_tok, c)
        out = attn.out_proj(out)  # "reduced" block: no residual, no FFN
        out = vis.post_layernorm(out)
        feats = self.model.visual_projection(out)[:, 1:]
        feats = F.normalize(feats.float(), dim=-1)
        n = CROP // PATCH
        return feats.view(b, n, n, -1)

    @torch.no_grad()
    def image_features(self, image: torch.Tensor) -> torch.Tensor:
        """[3,H,W] resized+normalised image -> [H/16, W/16, 512] window-averaged features."""
        _, height, width = image.shape
        crops, coords = [], []
        for y in range(0, height - CROP + 1, STRIDE):
            for x in range(0, width - CROP + 1, STRIDE):
                crops.append(image[:, y : y + CROP, x : x + CROP])
                coords.append((y // PATCH, x // PATCH))
        feats = self.crop_features(torch.stack(crops).to(self.device))
        n = CROP // PATCH
        acc = torch.zeros(height // PATCH, width // PATCH, feats.shape[-1], device=self.device)
        cnt = torch.zeros(height // PATCH, width // PATCH, 1, device=self.device)
        for (py, px), f in zip(coords, feats):
            acc[py : py + n, px : px + n] += f
            cnt[py : py + n, px : px + n] += 1
        return acc / cnt


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        height, width = resized_hw(img.width, img.height)
        img = img.resize((width, height), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) - CLIP_MEAN) / CLIP_STD
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), idx


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def extract_split(encoder: NaclipEncoder, images_dir: Path, ids: list[str], out_prefix: Path) -> None:
    """Write <prefix>_feats.npy and <prefix>_index.npz for the listed images."""
    paths = [Path(images_dir) / f"{i}.jpg" for i in ids]
    # Decoded in-process: containers may have a tiny /dev/shm for worker tensor sharing.
    loader = torch.utils.data.DataLoader(_ImageDataset(paths), batch_size=None, num_workers=0)
    feats: list[np.ndarray | None] = [None] * len(ids)
    for img, idx in loader:
        feats[int(idx)] = encoder.image_features(img).half().cpu().numpy()
        if int(idx) % 250 == 0:
            print(f"  {out_prefix.name}: {int(idx)}/{len(ids)}", flush=True)
    hw = np.array([f.shape[:2] for f in feats], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(hw[:, 0] * hw[:, 1])]).astype(np.int64)
    label_hw = hw * LABEL_SCALE
    label_offsets = np.concatenate([[0], np.cumsum(label_hw[:, 0] * label_hw[:, 1])]).astype(np.int64)
    flat = np.concatenate([f.reshape(-1, f.shape[-1]) for f in feats], axis=0)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{out_prefix}_feats.npy", flat)
    np.savez(
        f"{out_prefix}_index.npz",
        ids=np.array(ids),
        hw=hw,
        offsets=offsets,
        label_hw=label_hw,
        label_offsets=label_offsets,
    )
    print(f"  wrote {out_prefix}_feats.npy {flat.shape}", flush=True)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("data/downloads/clip-vit-base-patch16"))
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("text")
    t.add_argument("--out", type=Path, required=True)
    im = sub.add_parser("images")
    im.add_argument("--images-dir", type=Path, required=True)
    im.add_argument("--ids", type=Path, required=True)
    im.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    encoder = NaclipEncoder(args.model_dir, default_device())
    if args.cmd == "text":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out, encoder.encode_text(ADE20K_CLASSES, IMAGENET_TEMPLATES))
    else:
        extract_split(encoder, args.images_dir, read_ids(args.ids), args.out_prefix)


if __name__ == "__main__":
    main()
