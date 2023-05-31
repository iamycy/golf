from typing import Optional
import torch
from torch import Tensor
import torch.nn as nn
from torchaudio.transforms import MuLawEncoding, MuLawDecoding
from typing import Optional, Tuple


class ContinuousMuLawEncoding(MuLawEncoding):
    def forward(self, x: Tensor) -> Tensor:
        mu = self.quantization_channels - 1.0
        mu = torch.tensor(mu, dtype=x.dtype)
        x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
        x_mu = (x_mu + 1) / 2 * mu
        return x_mu


class ContinuousMuLawDecoding(MuLawDecoding):
    def forward(self, x_mu: Tensor) -> Tensor:
        mu = self.quantization_channels - 1.0
        mu = torch.tensor(mu, dtype=x_mu.dtype)
        x = (x_mu / mu) * 2 - 1
        x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1) / mu
        return x


class InterpolatedEmbedding(nn.Embedding):
    def forward(self, x: Tensor) -> Tensor:
        num_embeddings = self.num_embeddings
        assert x.dtype == torch.float32
        assert torch.all(x >= 0), x.min().item()
        assert torch.all(x <= (self.num_embeddings - 1)), x.max().item()

        raw_idx = x * (num_embeddings - 1)
        lower_idx = torch.floor(raw_idx).long().clip(0, num_embeddings - 2)
        upper_idx = lower_idx + 1
        p = raw_idx - lower_idx
        p = p.unsqueeze(-1)
        lower_embedding = super().forward(lower_idx)
        upper_embedding = super().forward(upper_idx)
        return lower_embedding * (1 - p) + upper_embedding * p


class SampleNet(nn.Module):
    def __init__(
        self,
        quantization_channels: int = 256,
        condition_channels: int = 128,
        a_channels: int = 192,
        b_channels: int = 32,
    ):
        super().__init__()

        self.a_channels = a_channels
        self.b_channels = b_channels

        self.embeddings = InterpolatedEmbedding(
            quantization_channels, quantization_channels
        )
        # for i in "urh":
        #     for j in "spe":
        #         self.register_module(
        #             f"embedding_{i}_{j}",
        #             InterpolatedEmbedding(quantization_channels, a_channels),
        #         )

        # self.W_a = nn.Linear(a_channels, a_channels * 3)
        # self.cond_linear = nn.Linear(condition_channels + 3 * quantization_channels, a_channels * 3, bias=False)
        self.gru_a = nn.GRU(
            condition_channels + 3 * quantization_channels,
            a_channels,
            bias=False,
            batch_first=True,
        )
        self.gru_b = nn.GRU(
            a_channels + condition_channels, b_channels, bias=False, batch_first=True
        )

        self.a = nn.Parameter(torch.randn(quantization_channels * 2))
        self.fc = nn.Sequential(
            nn.Linear(b_channels, quantization_channels * 2),
            nn.Tanh(),
        )

    def forward(self, f: Tensor, p: Tensor, s_prev: Tensor, e_prev: Tensor) -> Tensor:
        # GRU A
        # g_u, g_r, g_h = torch.chunk(self.cond_linear(f), 3, dim=-1)

        # u_s = self.embedding_u_s(s_prev)
        # u_p = self.embedding_u_p(p)
        # u_e = self.embedding_u_e(e_prev)
        # r_s = self.embedding_r_s(s_prev)
        # r_p = self.embedding_r_p(p)
        # r_e = self.embedding_r_e(e_prev)
        # h_s = self.embedding_h_s(s_prev)
        # h_p = self.embedding_h_p(p)
        # h_e = self.embedding_h_e(e_prev)

        # batch, T, _ = f.shape
        # h = f.new_zeros(batch, self.a_channels)
        # outputs = []
        # for i in range(T):
        #     h_u, h_r, h_h = torch.chunk(self.W_a(h), 3, dim=-1)
        #     u = torch.sigmoid(g_u[:, i] + h_u + u_s[:, i] + u_p[:, i] + u_e[:, i])
        #     r = torch.sigmoid(g_r[:, i] + h_r + r_s[:, i] + r_p[:, i] + r_e[:, i])
        #     h_hat = torch.tanh(g_h[:, i] + h_h * r + h_s[:, i] + h_p[:, i] + h_e[:, i])
        #     h = (1 - u) * h + u * h_hat
        #     outputs.append(h)

        # # GRU B
        # x = torch.stack(outputs, dim=1)
        # h = x.new_zeros(batch, self.b_channels)
        # outputs = []
        # for i in range(T):
        #     h = self.gru_b(x[:, i], h)
        #     outputs.append(h)

        # h = torch.stack(outputs, dim=1)

        p = self.embeddings(p)
        s = self.embeddings(s_prev)
        e = self.embeddings(e_prev)
        h = torch.cat([f, p, s, e], dim=-1)
        h, _ = self.gru_a(h)
        h = torch.cat([h, f], dim=-1)
        h, _ = self.gru_b(h)

        h = self.fc(h) * self.a
        h = h.view(h.shape[0], h.shape[1], -1, 2).sum(dim=-1)

        return h

    def sample_forward(
        self,
        f: Tensor,
        p: Tensor,
        s_prev: Tensor,
        e_prev: Tensor,
        states: Optional[Tuple[Tensor, ...]] = None,
    ):

        batch, _ = f.shape

        if states is None:
            state_a = f.new_zeros(batch, self.a_channels)
            state_b = f.new_zeros(batch, self.b_channels)
        else:
            state_a, state_b = states

        g_u, g_r, g_h = torch.chunk(self.cond_linear(f), 3, dim=-1)

        u_s = self.embedding_u_s(s_prev)
        u_p = self.embedding_u_p(p)
        u_e = self.embedding_u_e(e_prev)
        r_s = self.embedding_r_s(s_prev)
        r_p = self.embedding_r_p(p)
        r_e = self.embedding_r_e(e_prev)
        h_s = self.embedding_h_s(s_prev)
        h_p = self.embedding_h_p(p)
        h_e = self.embedding_h_e(e_prev)

        h_u, h_r, h_h = torch.chunk(self.W_a(state_a), 3, dim=-1)
        u = torch.sigmoid(g_u + h_u + u_s + u_p + u_e)
        r = torch.sigmoid(g_r + h_r + r_s + r_p + r_e)
        h_hat = torch.tanh(g_h + h_h * r + h_s + h_p + h_e)
        state_a = (1 - u) * state_a + u * h_hat

        state_b = self.gru_b(state_a, state_b)

        h = self.fc(state_b) * self.a
        h = h.view(batch, -1, 2).sum(dim=-1)

        return h, (state_a, state_b)
