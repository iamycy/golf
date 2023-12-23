import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram
from typing import Optional, Tuple, Union, List, Callable, Any

from models.utils import get_window_fn, AudioTensor


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss.
    """

    eps = 1e-8

    def __init__(self, alpha: float = 1.0, window: str = "hann", **kwargs):
        super().__init__()
        self.alpha = alpha
        self.spec = Spectrogram(power=1, window_fn=get_window_fn(window), **kwargs)

    def forward(self, pred: AudioTensor, target: AudioTensor):
        S_true = self.spec(target.as_tensor())
        S_pred = self.spec(pred.as_tensor())
        linear_term = F.l1_loss(S_pred, S_true)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())
        loss = linear_term + self.alpha * log_term
        return loss


class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)
    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(
        self,
        n_ffts: list,
        alpha=1.0,
        ratio=1.0,
        overlap=0.75,
        **kwargs,
    ):
        super().__init__()
        self.losses = nn.ModuleList(
            [
                SSSLoss(
                    alpha=alpha,
                    n_fft=n_fft,
                    hop_length=int(n_fft - n_fft * overlap),
                    **kwargs,
                )
                for n_fft in n_ffts
            ]
        )
        self.ratio = ratio

    def forward(self, x_pred: AudioTensor, x_true: AudioTensor):
        return self.ratio * sum(loss(x_pred, x_true) for loss in self.losses)


class MSSLossV2(nn.Module):
    """
    Multi-scale Spectral Loss (revisited)
    """

    def __init__(
        self,
        n_ffts: List[int],
        distance: Union[nn.L1Loss, nn.MSELoss],
        compression: str = "log1p",
        window: str = "hann",
        overlap=0.75,
        ratio=1.0,
        **kwargs,
    ):
        super().__init__()

        self.distance = distance
        self.ratio = ratio

        if compression == "log1p":
            self.compress = torch.log1p
        elif compression == "log":
            self.compress = lambda x: torch.log(x + 1e-7)
        elif compression == "id":
            self.compress = lambda x: x
        else:
            raise ValueError(f"Unknown compression: {compression}")
        window_fn = get_window_fn(window)

        self.specs = nn.ModuleList(
            [
                Spectrogram(
                    n_fft=n_fft,
                    hop_length=int(n_fft - n_fft * overlap),
                    window_fn=window_fn,
                    power=1,
                    **kwargs,
                )
                for n_fft in n_ffts
            ]
        )

    def forward(self, x_pred: AudioTensor, x_true: AudioTensor):
        return self.ratio * sum(
            self.distance(
                self.compress(spec(x_pred.as_tensor())),
                self.compress(spec(x_true.as_tensor())),
            )
            for spec in self.specs
        )
