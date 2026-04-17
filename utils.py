from __future__ import annotations
from typing import Sequence, Callable

import torch
from torch import Tensor
import torch.nn.functional as F
import deepinv as dinv
from deepinv.physics import Physics, Inpainting, Demosaicing
from deepinv.physics import PhysicsCropper
from deepinv.utils.tensorlist import TensorList

def get_4x4_mask(x: Tensor) -> Tensor:
    """Get 4x4 (16-band) sequential multispectral filter array mask.

    :param torch.Tensor x: image BCHW for which shape of mask is calculated.
    :return: torch.Tensor filter mask.
    """

    mask = torch.zeros_like(x)
    for i in range(4):
        for j in range(4):
            c = i * 4 + j    
            if c < x.shape[1]: # only assign if exists
                mask[:, c, i::4, j::4] = 1

    return mask.int()

def demosaic_gaussian(y: Tensor, physics: Inpainting | Demosaicing) -> Tensor:
    """Demosaic using Gaussian-like kernel.

    :param torch.Tensor y: input tensor (unflattened) of shape BCHW
    :param deepinv.physics.Inpainting physics: physics containing mask to use.
    :return: torch.Tensor demosaiced image of shape BCHW
    """
    out = torch.zeros_like(y)
    mask = physics.mask.float()

    kernel_size = 9
    pad = [kernel_size // 2] * 4
    
    # Gaussian-like kernel
    K = torch.ones(1, 1, kernel_size, kernel_size, device=y.device)
    
    center = kernel_size // 2
    for ki in range(kernel_size):
        for kj in range(kernel_size):
            dist = ((ki - center)**2 + (kj - center)**2)**0.5
            K[0, 0, ki, kj] = max(0, 1.0 - dist / (kernel_size / 2))
    
    K /= K.sum()

    for c in range(y.shape[1]):
        y_c    = y   [:, c:c+1, :, :]
        mask_c = mask[:, c:c+1, :, :]

        y_c_pad    = F.pad(y_c   , pad, mode='replicate')
        mask_c_pad = F.pad(mask_c, pad, mode='replicate')

        out[:, c:c+1, :, :] = torch.where(mask_c > 0, y_c, F.conv2d(y_c_pad, K) / (F.conv2d(mask_c_pad, K) + 1e-8))
    
    return out


class RAM_init_hijack(dinv.models.RAM):
    """Identical to deepinv.models.RAM except model input is custom function instead of adjoint.

    :param Callable custom_init: function f where x_in = f(y, physics) where x_in is the input to the network and y is the measurement.
    """
    def __init__(self, custom_init: Callable[[Tensor, Physics], Tensor]=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_init = custom_init

    def forward(
        self,
        y: torch.Tensor,
        physics: Physics = None,
        sigma: float | Tensor = None,
        gain: float | Tensor = None,
        img_size: Sequence[int] = None,
    ) -> torch.Tensor:
        r"""
        Identical to deepinv.models.RAM except model input is custom function instead of adjoint.
        """
        if physics is None and sigma is None and gain is None:
            raise ValueError(
                "Either physics, sigma or gain must be provided to the RAM model."
            )

        if isinstance(y, TensorList):
            max_val = y[0].abs().reshape(y[0].size(0), -1).amax(dim=1, keepdim=False)
        else:
            max_val = y.abs().reshape(y.size(0), -1).amax(dim=1, keepdim=False)

        # rescale elements in the batch where max_val > 5 * sigma_threshold
        rescale_val = torch.where(
            max_val > 5 * self.sigma_threshold,
            torch.tensor(1.0, device=max_val.device, dtype=max_val.dtype),
            max_val,
        )

        if isinstance(y, TensorList):
            for yi in y:
                yi /= rescale_val.view([yi.shape[0]] + [1] * (yi.ndim - 1))
        else:
            y = y / rescale_val.view([y.shape[0]] + [1] * (y.ndim - 1))

        if physics is None:
            physics = dinv.physics.Denoising(noise_model=dinv.physics.ZeroNoise())

        if img_size is None:
            if hasattr(physics, "img_shape") and physics.img_shape is not None:
                img_size = physics.img_shape
            elif hasattr(physics, "img_size") and physics.img_size is not None:
                img_size = physics.img_size
            else:
                img_size = physics.A_adjoint(y).shape[1:]

        sigma, gain = self.obtain_sigma_gain(
            physics=physics,
            sigma=sigma,
            gain=gain,
            rescale_val=rescale_val,
            device=y.device,
        )

        pad = (-img_size[-2] % 8, -img_size[-1] % 8)

        use_pad = False
        if pad[0] != 0 or pad[1] != 0:
            physics = PhysicsCropper(physics, pad)
            use_pad = True

        x_in = self.custom_init(y, physics)

        sigma = torch.maximum(
            sigma, torch.tensor(self.sigma_threshold, device=x_in.device)
        )
        sigma = self._handle_sigma(sigma)

        gain = torch.maximum(
            gain, torch.tensor(self.gain_threshold, device=x_in.device)
        )
        gain = self._handle_sigma(gain)

        out = self.forward_unet(x_in, sigma=sigma, gain=gain, physics=physics, y=y)

        if use_pad:
            out = physics.remove_pad(out)

        out = out * rescale_val.view([out.shape[0]] + [1] * (out.ndim - 1))

        return out
