import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List

def make_mlp(in_dim, out_dim, hidden_dim=None, activation=nn.ReLU(), final_activation=None):
    if hidden_dim is None:
        hidden_dim = in_dim
    layers = [
        nn.Linear(in_dim, hidden_dim),
        activation,
        nn.Linear(hidden_dim, hidden_dim),
        activation,
        nn.Linear(hidden_dim, out_dim)
    ]
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class ZoneoutLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, zoneout_prob: float = 0.1):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.zoneout_prob = float(zoneout_prob)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state  # [B, hidden]
        h_new, c_new = self.cell(x, (h, c))

        if self.training and self.zoneout_prob > 0.0:
            # With probability zoneout_prob, keep old state instead of new
            keep_mask_h = (torch.rand_like(h) < self.zoneout_prob).to(h.dtype)
            keep_mask_c = (torch.rand_like(c) < self.zoneout_prob).to(c.dtype)

            h = keep_mask_h * h + (1.0 - keep_mask_h) * h_new
            c = keep_mask_c * c + (1.0 - keep_mask_c) * c_new
        else:
            h, c = h_new, c_new

        return h, c


# ------------------------
# Zoneout LSTM
# ------------------------
class ZoneoutLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        zoneout_prob: float = 0.1,
        batch_first: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            in_size = input_size if layer == 0 else hidden_size
            self.layers.append(ZoneoutLSTMCell(in_size, hidden_size, zoneout_prob))

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        hx: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            x = x.transpose(0, 1)  # [seq_len, batch, input]

        seq_len, batch_size, _ = x.size()

        if hx is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                             device=x.device, dtype=x.dtype)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                             device=x.device, dtype=x.dtype)
            hx = (h0, c0)
        else:
            hx = (hx[0].to(x.device).type_as(x), hx[1].to(x.device).type_as(x))

        outputs = []
        hs, cs = list(hx[0]), list(hx[1])

        for t in range(seq_len):
            inp = x[t]
            new_hs, new_cs = [], []
            for l, layer in enumerate(self.layers):
                h, c = layer(inp, (hs[l], cs[l]))
                inp = h
                if l < self.num_layers - 1:  # apply dropout between layers
                    inp = self.dropout_layer(inp)
                new_hs.append(h)
                new_cs.append(c)
            hs, cs = new_hs, new_cs
            outputs.append(inp)

        outputs = torch.stack(outputs, dim=0)  # [seq_len, batch, hidden]

        h_n = torch.stack(hs, dim=0)  # [num_layers, batch, hidden]
        c_n = torch.stack(cs, dim=0)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # [batch, seq_len, hidden]

        return outputs, (h_n, c_n)


# ------------------------
# Utility: Positional encoding
# ------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # [B, T, d_model]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class ConvPrenet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=5, kernel_size=5, dropout=0.05):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)

        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    hidden_dim, hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
        self.convs = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = x.transpose(1, 2)

        for conv in self.convs:
            x = F.relu(conv(x))
            x = self.dropout(x)
        x = x.transpose(1, 2)
        return x

class DecoderPreNet(nn.Module):
    def __init__(self, in_dim, hidden_sizes=[256, 256], p_drop=0.5):
        super().__init__()
        layers, cur = [], in_dim
        for hdim in hidden_sizes:
            layers += [nn.Linear(cur, hdim), nn.LeakyReLU(0.1), nn.Dropout(p_drop)]
            cur = hdim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MultiHeadLSA(nn.Module):
    def __init__(self, d_query, d_key, d_model, d_attn, num_heads,
                 drop_out=0.1, conv_channels=32, conv_kernel=31):
        super().__init__()
        self.num_heads = num_heads
        self.d_attn = d_attn
        self.dropout = nn.Dropout(drop_out) if drop_out > 0 else nn.Identity()
        self.query_proj = nn.ModuleList([nn.Linear(d_query, d_attn) for _ in range(num_heads)])
        self.key_proj   = nn.ModuleList([nn.Linear(d_key, d_attn) for _ in range(num_heads)])

        self.loc_proj   = nn.ModuleList([
            nn.Conv1d(2, conv_channels, kernel_size=conv_kernel, padding=conv_kernel // 2)
            for _ in range(num_heads)
        ])
        self.loc_linear = nn.ModuleList([nn.Linear(conv_channels, d_attn) for _ in range(num_heads)])

        self.v = nn.ModuleList([nn.Linear(d_attn, 1, bias=True) for _ in range(num_heads)])

        self.out_proj = nn.Linear(num_heads * d_key, d_model)
        self.out_attn_proj = nn.Linear(num_heads, 1)

        for layer in self.v:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, query, keys, prev_attn, cum_attn, pad_mask=None):
        """
        query: [B, d_query]
        keys:  [B, T, d_key]
        prev_attn, cum_attn: [B, T]
        pad_mask: [B, T] boolean mask
        """
        all_contexts, all_weights = [], []
        scale = math.sqrt(self.d_attn)
        for h in range(self.num_heads):
            q = self.query_proj[h](query).unsqueeze(1)  # [B,1,d_attn]
            k = self.key_proj[h](keys)                  # [B,T,d_attn]

            loc_input = torch.stack([prev_attn, cum_attn], dim=1)  # [B,2,T]
            f = self.loc_proj[h](loc_input)                        # [B,C,T]
            f = f.transpose(1,2)                                   # [B,T,C]
            f = self.loc_linear[h](f)                               # [B,T,d_attn]

            e = self.v[h](torch.tanh((q + k + f))).squeeze(-1) / scale

            if pad_mask is not None:
                e = e.masked_fill(pad_mask, -1e9)

            alpha = torch.softmax(e, dim=-1)

            context = torch.bmm(alpha.unsqueeze(1), keys)
            all_contexts.append(context)
            all_weights.append(alpha)

        context = torch.cat(all_contexts, dim=-1)
        context = self.out_proj(context).squeeze(1)
        attn_weights = self.out_attn_proj(torch.stack(all_weights, dim=-1)).squeeze(-1)
        return context, attn_weights

class MetaGate(nn.Module):
    def __init__(self, hidden_dim, meta_dim):
        super().__init__()
        self.fc_gamma = nn.Linear(meta_dim, hidden_dim)
        self.fc_beta = nn.Linear(meta_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, h, meta):
        gamma = torch.tanh(self.fc_gamma(meta))
        beta = torch.tanh(self.fc_beta(meta))
        out = h * (1 + gamma) + beta
        return self.ln(out)

class OsuModel(nn.Module):
    def __init__(self, input_dim, d_model, num_types, meta_dim, num_layer=4):
        super().__init__()
        self.d_model = d_model
        self.num_types = num_types
        self.num_layer = num_layer

        self.prenet = ConvPrenet(input_dim, d_model)
        self.posenc = SinusoidalPositionalEncoding(d_model)
        self.dec_prenet = DecoderPreNet(d_model)
        self.meta_gate = MetaGate(d_model * 2, meta_dim)

        self.emb_x = nn.Linear(1, d_model)
        self.emb_y = nn.Linear(1, d_model)
        self.emb_type = nn.Embedding(num_types, d_model)
        self.emb_dt = nn.Linear(1, d_model)
        self.emb_proj = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)

        self.stops = make_mlp(d_model, 1, hidden_dim=d_model, activation=nn.ReLU(), final_activation=nn.Sigmoid())
        self.out_x = make_mlp(d_model, 1, hidden_dim=d_model, activation=nn.ReLU(), final_activation=nn.Sigmoid())
        self.out_y = make_mlp(d_model, 1, hidden_dim=d_model, activation=nn.ReLU(), final_activation=nn.Sigmoid())
        self.out_dt = make_mlp(d_model, 1, hidden_dim=d_model, activation=nn.ReLU())
        self.out_type = make_mlp(d_model, num_types, hidden_dim=d_model, activation=nn.ReLU())

        self.attention = MultiHeadLSA(d_model, d_model, d_model, d_model, num_heads=2)
        self.attn_rnn = ZoneoutLSTM(d_model + 256, d_model, batch_first=True, num_layers=num_layer)
        self.decoder_rnn = ZoneoutLSTM(d_model * 2, d_model, batch_first=True, num_layers=num_layer)

    def embed_truth(self, truth_step):
        x_emb = self.emb_x(truth_step[:, 0].float().unsqueeze(1))
        y_emb = self.emb_y(truth_step[:, 1].float().unsqueeze(1))
        type_emb = self.emb_type(truth_step[:, 3].long())
        dt_emb = self.emb_dt(truth_step[:, 2].float().unsqueeze(1))
        if x_emb.dim() == 1:
            x_emb, y_emb, dt_emb = x_emb.unsqueeze(0), y_emb.unsqueeze(0), dt_emb.unsqueeze(0)

        _, (h, c) = self.emb_proj(torch.stack((x_emb, y_emb, type_emb, dt_emb), dim=1))
        return h.squeeze(0)

    def forward(self, inputs, truth=None, pad_mask=None,
                meta=None, steps=None, teacher_forcing_ratio=1.0):
        B, T, _ = inputs.shape
        device = inputs.device
        if steps is None and truth is not None:
            steps = truth.size(1)
        elif steps is None:
            steps = T

        latent_seq = self.prenet(inputs)
        latent_seq = self.posenc(latent_seq)

        prev_attn = torch.zeros((B, latent_seq.size(1)), device=device)
        prev_attn[:, 0] = 1.
        cum_attn = prev_attn.clone()

        attn_h = torch.zeros(self.num_layer, B, self.d_model, device=device)
        attn_c = attn_h.clone()

        dec_h  = torch.zeros(self.num_layer, B, self.d_model, device=device)
        dec_c = dec_h.clone()
        context = torch.zeros(B, self.d_model, device=device)

        prev_embed = torch.zeros(B, self.d_model, device=device)

        outs_x, outs_y, outs_dt, outs_type, stops, alignments = [], [], [], [], [], []
        for i in range(steps):
            pm = self.dec_prenet(prev_embed)
            _, (attn_h, attn_c) = self.attn_rnn(torch.cat([pm, context], -1).unsqueeze(1), (attn_h, attn_c))

            context, attn_weights = self.attention(attn_h[-1], latent_seq, prev_attn, cum_attn, pad_mask)

            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

            cum_attn = cum_attn + attn_weights
            cum_attn = cum_attn / cum_attn.sum(dim=1, keepdim=True)

            prev_attn = attn_weights

            step_input = torch.cat([attn_h[-1], context], dim=-1).unsqueeze(1)
            step_gate = self.meta_gate(step_input, meta)
            _, (dec_h, dec_c) = self.decoder_rnn(step_gate, (dec_h, dec_c))

            hf = dec_h.mean(0)

            x_logits = self.out_x(hf).squeeze(-1)
            y_logits = self.out_y(hf).squeeze(-1)
            dt = F.relu(self.out_dt(hf)).squeeze(-1)
            type_logits = self.out_type(hf)
            stop_logits = self.stops(hf).squeeze(-1)

            outs_x.append(x_logits)
            outs_y.append(y_logits)
            outs_dt.append(dt)
            outs_type.append(type_logits)
            stops.append(stop_logits)
            alignments.append(attn_weights)

            use_tf = (truth is not None) and (torch.rand(1, device=device).item() < teacher_forcing_ratio)
            if use_tf:
                next_embed = self.embed_truth(truth[:, i]).detach()
            else:
                x_emb = self.emb_x(x_logits)
                y_emb = self.emb_y(y_logits)
                type_emb = self.emb_type(type_logits.argmax(-1))
                dt_emb = self.emb_dt(dt.unsqueeze(1))
                _, (h, c) = self.emb_proj(torch.stack((x_emb, y_emb, type_emb, dt_emb), dim=1))
                next_embed = h.squeeze(1)

            prev_embed = next_embed.detach()

        return (torch.stack(outs_x, dim=1),
                torch.stack(outs_y, dim=1),
                torch.stack(outs_dt, dim=1),
                torch.stack(outs_type, dim=1),
                torch.stack(stops, dim=1).squeeze(-1),
                torch.stack(alignments, dim=1))
