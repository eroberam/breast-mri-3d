"""Small residual 3D network with anatomical and lesion decoders."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, source: int, target: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(source, target, 3, padding=1, bias=False), nn.GroupNorm(4, target), nn.SiLU(),
            nn.Conv3d(target, target, 3, padding=1, bias=False), nn.GroupNorm(4, target),
        )
        self.skip = nn.Identity() if source == target else nn.Conv3d(source, target, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(x) + self.skip(x))


class Decode(nn.Module):
    def __init__(self, widths: tuple[int, ...]) -> None:
        super().__init__()
        pairs = list(zip(widths[1:], widths[:-1], strict=True))[::-1]
        self.up = nn.ModuleList(nn.ConvTranspose3d(a, b, 2, stride=2) for a, b in pairs)
        self.fuse = nn.ModuleList(Residual(2 * b, b) for _, b in pairs)

    def forward(self, x: torch.Tensor, skip: list[torch.Tensor]) -> torch.Tensor:
        for up, fuse, layer in zip(self.up, self.fuse, reversed(skip), strict=True):
            x = fuse(torch.cat((up(x), layer), dim=1))
        return x


class BreastMRI3D(nn.Module):
    def __init__(self, width: int = 12, *, dual_decoder: bool = True, input_mode: str = "dce") -> None:
        super().__init__()
        if width < 4 or width % 4:
            raise ValueError("The first-stage width must be a positive multiple of four.")
        widths = (width, width * 2, width * 4, width * 6)
        self.width = width
        self.dual_decoder = dual_decoder
        if input_mode not in {"dce", "subtraction"}:
            raise ValueError("Use DCE or subtraction inputs.")
        self.input_mode = input_mode
        self.first = Residual(3, widths[0])
        self.encoder = nn.ModuleList(nn.Sequential(nn.Conv3d(a, b, 2, stride=2), Residual(b, b))
                                     for a, b in zip(widths[:-1], widths[1:], strict=True))
        self.anatomy_decoder = Decode(widths)
        self.lesion_decoder = Decode(widths) if dual_decoder else None
        self.anatomy_head = nn.Conv3d(width, 4, 1)
        self.lesion_head = nn.Conv3d(width, 1, 1)
        self.ranking_head = nn.Conv3d(width, 1, 1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if x.ndim != 5 or x.shape[1] != 3 or any(n % 8 for n in x.shape[-3:]):
            raise ValueError("Input must be N×3×Z×Y×X with spatial dimensions divisible by eight.")
        if self.input_mode == "subtraction":
            x = torch.cat((torch.zeros_like(x[:, :2]), x[:, 2:]), dim=1)
        x = self.first(x)
        keep = []
        for stage in self.encoder:
            keep.append(x)
            x = stage(x)
        return x, keep

    def lesion(self, x: torch.Tensor) -> torch.Tensor:
        encoded, skip = self._encode(x)
        decoder = self.lesion_decoder if self.lesion_decoder is not None else self.anatomy_decoder
        return self.lesion_head(decoder(encoded, skip))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, keep = self._encode(x)
        anatomy = self.anatomy_decoder(x, keep)
        lesion = self.lesion_decoder(x, keep) if self.lesion_decoder is not None else anatomy
        return torch.cat((self.anatomy_head(anatomy), self.lesion_head(lesion)), 1), self.ranking_head(lesion)
