import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchvision.transforms import v2 as transforms
import numpy as np

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet

import math


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        self.use_clstoken = use_clstoken
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
        )

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        return out


class DepthAnythingV2(nn.Module):
    def __init__(
        self,
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024],
        use_bn=False,
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.rgb_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)
        depth = self.depth_head(features, patch_h, patch_w)
        return depth.squeeze(1)

    @torch.no_grad()
    def infer_image(self, raw_images, w=1920, h=1080, interpolate_mode="bicubic", flip_aug=False, with_pad=False, pad_mode="constant"):
        if not isinstance(raw_images, (torch.Tensor, np.ndarray)):
            raise TypeError("raw_images must be a single image (torch.Tensor or numpy.ndarray) or a batch tensor.")
        images, meta = self.image2tensor(raw_images, w, h, with_pad, pad_mode)
        orig_h, orig_w = meta[0], meta[1]
        pad_t, pad_l = (meta[2], meta[3]) if len(meta) == 4 else (0, 0)
        if flip_aug:
            flipped_images = torch.flip(images, dims=[3])
            depths_normal = self.forward(images)
            depths_flipped = self.forward(flipped_images)
            depths_flipped = torch.flip(depths_flipped, dims=[2])
            depths = (depths_normal + depths_flipped) / 2.0
        else:
            depths = self.forward(images)
        if with_pad:
            depths = depths[:, pad_t:pad_t + orig_h, pad_l:pad_l + orig_w]
            return depths if depths.shape[0] > 1 else depths.squeeze(0)
        batch_size = images.shape[0]
        interpolated_depths = F.interpolate(depths.unsqueeze(1), size=(orig_h, orig_w), mode="nearest-exact").squeeze(1)
        if batch_size == 1:
            return interpolated_depths.squeeze(0)
        return interpolated_depths

    def image2tensor(self, raw_image, w=1920, h=1080, with_pad=False, pad_mode="constant"):
        if isinstance(raw_image, torch.Tensor):
            if raw_image.ndim == 3:
                raw_image = raw_image.unsqueeze(0)
            orig_H, orig_W = raw_image.shape[2:]
            if with_pad:
                tgt_H = max(h, math.ceil(orig_H / 14) * 14)
                tgt_W = max(w, math.ceil(orig_W / 14) * 14)
                diff_H = tgt_H - orig_H
                diff_W = tgt_W - orig_W
                pad_t = diff_H // 2
                pad_b = diff_H - pad_t
                pad_l = diff_W // 2
                pad_r = diff_W - pad_l
                mode = "reflect" if pad_mode == "reflect" else "constant"
                image = F.pad(raw_image, (pad_l, pad_r, pad_t, pad_b), mode=mode)
                image = self.rgb_transform(image)
                image = image.to(raw_image.device)
                return image, (orig_H, orig_W, pad_t, pad_l)
            new_H, new_W = 1078, 1918
            image = F.interpolate(raw_image, size=(new_H, new_W), mode="bicubic", align_corners=True)
            image = torch.clamp(image, 0, 1)
            image = self.rgb_transform(image)
            image = image.to(raw_image.device)
            return image, (orig_H, orig_W)
        else:
            orig_H, orig_W = raw_image.shape[:2]
            image_np = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            transform = Compose(
                [
                    Resize(
                        width=w,
                        height=h,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method="lower_bound",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    PrepareForNet(),
                ]
            )
            processed = transform({"image": image_np})
            image = torch.from_numpy(processed["image"]).unsqueeze(0)
            return image, (orig_H, orig_W)
