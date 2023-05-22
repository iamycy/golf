import torch
from torch import Tensor
from pysptk.synthesis import Synthesizer, AllPoleDF
import numpy as np

from models import filters
from models.utils import AudioTensor


class SampleBasedLTVMinimumPhaseFilter(filters.LTVMinimumPhaseFilter):
    def forward(self, ex: AudioTensor, gain: AudioTensor, a: AudioTensor):
        assert ex.ndim == 2
        assert gain.ndim == 2
        assert a.ndim == 3
        assert a.shape[1] == gain.shape[1]
        device = ex.device
        dtype = ex.dtype
        hop_length = gain.hop_length

        ex = ex.as_tensor()
        gain = gain.as_tensor()
        a = a.as_tensor()

        # one sample at a time
        assert ex.shape[0] == 1
        ex = ex.squeeze(0).cpu().numpy().astype(np.float64)
        gain = gain.squeeze(0).cpu().numpy().astype(np.float64)
        a = a.squeeze(0).cpu().numpy().astype(np.float64)

        order = a.shape[-1]

        synthesizer = Synthesizer(AllPoleDF(order=order), hop_length)
        lpc_coeffs = np.concatenate([np.log(gain[:, None]), a], axis=1)

        y = synthesizer.synthesis(ex, lpc_coeffs)
        return AudioTensor(torch.from_numpy(y).to(device).to(dtype).unsqueeze(0))


def convert2samplewise(config: dict):
    for key, value in config.items():
        if key == "class_path":
            if ".LTVMinimumPhaseFilter" in config["class_path"]:
                config["class_path"] = "infer.SampleBasedLTVMinimumPhaseFilter"
                return config
            elif ".LTVMinimumPhaseFIRFilter" in config["class_path"]:
                config["class_path"] = "models.filters.LTVMinimumPhaseFIRFilterPrecise"
                config["init_args"].pop("conv_method")
                return config
            elif ".LTVZeroPhaseFIRFilter" in config["class_path"]:
                config["class_path"] = "models.filters.LTVZeroPhaseFIRFilterPrecise"
                config["init_args"].pop("conv_method")
                return config
        elif isinstance(value, dict):
            config[key] = convert2samplewise(value)
    return config
