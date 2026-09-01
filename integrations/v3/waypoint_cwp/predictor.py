"""Clean inference wrapper around SmartWay's enhanced CWP waypoint predictor.

Vendored from Reproductions/SmartWay-Code (itself based on
Discrete-Continuous-VLN, Hong et al.).  Frozen weights, inference only.

Input convention (matches SmartWay's clockwise slot layout):
  slot i (i=0..11) is the view after rotating the agent *right* by 30*i deg
  (slot 0 = current forward view).  RGB HxWx3 uint8 (224x224 native),
  depth 256x256 float in [0,1] (= meters / 10, clipped).

Output: up to `max_predictions` waypoints as relative heading in radians
(CCW-positive, [-pi, pi), 0 = forward) + distance (0.25..3.0 m) + score,
plus the full 120x12 softmax heatmap and angle-entropy / top1-prob
uncertainty signals (look-around trigger candidates).
"""
import math
import os

import numpy as np
import torch

SMARTWAY_ROOT = "/data/pengyh/workspace/Reproductions/SmartWay-Code"
DEFAULT_CKPT = os.path.join(SMARTWAY_ROOT, "waypoint_predictor/checkpoints/final-camera-ready")
DEFAULT_DDPPO = os.path.join(
    SMARTWAY_ROOT, "data/pretrained_models/ddppo-models/gibson-2plus-resnet50.pth")
DEFAULT_TORCH_HOME = "/data1/cache/torch"
DEFAULT_HF_CACHE = "/data1/cache/huggingface/hf-hub"

NUM_IMGS = 12
NUM_ANGLES = 120
NUM_CLASSES = 12


class CWPPredictor:
    def __init__(self, ckpt=DEFAULT_CKPT, ddppo_ckpt=DEFAULT_DDPPO, device="cuda",
                 max_predictions=5, nms_sigma=(7.0, 5.0)):
        os.environ.setdefault("TORCH_HOME", DEFAULT_TORCH_HOME)
        os.environ.setdefault("HF_HUB_CACHE", DEFAULT_HF_CACHE)
        from transformers import AutoImageProcessor

        from .TRM_net import BinaryDistPredictor_TRM
        from .ddppo_resnet.resnet_policy import PNResnetDepthEncoder

        self.device = torch.device(device)
        self.max_predictions = max_predictions
        self.nms_sigma = nms_sigma

        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        hub_repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dinov2_hub")
        self.dino = torch.hub.load(hub_repo, "dinov2_vits14_reg", source="local",
                                   pretrained=True).to(self.device).eval()

        self.depth_encoder = PNResnetDepthEncoder()
        ddppo = torch.load(ddppo_ckpt, map_location="cpu", weights_only=False)
        weights = {}
        for k, v in ddppo["state_dict"].items():
            parts = k.split(".")[2:]
            if parts and parts[0] == "visual_encoder":
                weights[".".join(parts[1:])] = v
        self.depth_encoder.load_state_dict(weights, strict=True)
        self.depth_encoder.to(self.device).eval()

        self.predictor = BinaryDistPredictor_TRM()
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state["predictor"]["state_dict"]
        missing, unexpected = self.predictor.load_state_dict(state, strict=False)
        # SmartWay itself loads with strict=False; surface what got skipped once.
        self.load_report = {"missing": list(missing), "unexpected": list(unexpected)}
        self.predictor.to(self.device).eval()

        for module in (self.dino, self.depth_encoder, self.predictor):
            for p in module.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, rgb_clockwise, depth_clockwise):
        from . import wp_utils

        assert len(rgb_clockwise) == NUM_IMGS and len(depth_clockwise) == NUM_IMGS
        rgb = torch.stack([
            torch.from_numpy(np.ascontiguousarray(r[..., :3], dtype=np.uint8))
            for r in rgb_clockwise])                                   # (12,H,W,3)
        pix = self.processor(images=rgb.permute(0, 3, 1, 2),
                             return_tensors="pt")["pixel_values"].to(self.device)
        rgb_feats = self.dino(pix)                                     # (12,384)

        depth = np.stack([
            np.asarray(d, dtype=np.float32).reshape(256, 256, 1)
            for d in depth_clockwise])
        depth_t = torch.from_numpy(np.clip(depth, 0.0, 1.0)).to(self.device)
        depth_feats = self.depth_encoder(depth_t)                      # (12,128,4,4)

        logits = self.predictor(rgb_feats, depth_feats, None,
                                pre_fuse=False, cross_attn=True)       # (1,120,12)
        probs = torch.softmax(logits.reshape(1, -1), dim=1).reshape(1, NUM_ANGLES, NUM_CLASSES)
        wrapped = torch.cat((probs[:, -1:, :], probs, probs[:, :1, :]), dim=1)
        nms_map = wp_utils.nms(wrapped.unsqueeze(1), max_predictions=self.max_predictions,
                               sigma=self.nms_sigma).squeeze(1)[:, 1:-1, :][0]

        nz = nms_map.nonzero()
        waypoints = []
        for a_idx, d_idx in nz.tolist():
            heading = -(a_idx / NUM_ANGLES) * 2 * math.pi              # clockwise idx -> CCW rad
            heading = (heading + math.pi) % (2 * math.pi) - math.pi
            waypoints.append({
                "heading_rad": heading,
                "heading_deg": math.degrees(heading),
                "distance_m": (d_idx + 1) * 0.25,
                "score": float(nms_map[a_idx, d_idx]),
            })
        waypoints.sort(key=lambda w: -w["score"])

        angle_marginal = probs[0].sum(-1)
        entropy = float(-(angle_marginal * torch.log(angle_marginal + 1e-12)).sum())
        return {
            "waypoints": waypoints,
            "heatmap": probs[0].cpu().numpy(),
            "sigmoid": torch.sigmoid(logits)[0].cpu().numpy(),
            "angle_entropy": entropy,
            "angle_top1": float(angle_marginal.max()),
        }
