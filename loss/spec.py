import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram

from models.utils import get_window_fn


class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss.
    """

    eps = 1e-8

    def __init__(self, alpha: float = 1.0, window: str = "hann", **kwargs):
        super().__init__()
        self.alpha = alpha
        self.spec = Spectrogram(power=1, window_fn=get_window_fn(window), **kwargs)

    def forward(self, pred, target):
        S_true = self.spec(target)
        S_pred = self.spec(pred)
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

    def forward(self, x_pred, x_true):
        return self.ratio * sum(loss(x_pred, x_true) for loss in self.losses)
