# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.models import DOFA_CLIP, dofa_clip_base_patch16_224


class TestDOFA_CLIP:
    def test_dofa_clip(self) -> None:
        model = DOFA_CLIP(
            embed_dim=128,
            image_embed_dim=384,
            image_depth=2,
            image_num_heads=6,
            text_width=128,
            text_layers=2,
            text_heads=8,
        )
        image = torch.rand(2, 4, 224, 224)
        text = torch.randint(0, 100, (3, 77), dtype=torch.long)
        wavelengths = [0.665, 0.56, 0.49, 0.842]
        logits_per_image, logits_per_text = model(image, text, wavelengths)
        assert logits_per_image.shape == torch.Size([2, 3])
        assert logits_per_text.shape == torch.Size([3, 2])

    def test_dofa_clip_builder(self) -> None:
        model = dofa_clip_base_patch16_224(
            embed_dim=128,
            image_embed_dim=384,
            image_depth=2,
            image_num_heads=6,
            text_width=128,
            text_layers=2,
            text_heads=8,
        )
        image = torch.rand(1, 4, 224, 224)
        text = torch.randint(0, 100, (1, 77), dtype=torch.long)
        wavelengths = [0.665, 0.56, 0.49, 0.842]
        logits_per_image, logits_per_text = model(image, text, wavelengths)
        assert logits_per_image.shape == torch.Size([1, 1])
        assert logits_per_text.shape == torch.Size([1, 1])
