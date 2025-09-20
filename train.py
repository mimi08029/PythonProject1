import os
import matplotlib.pyplot as plt
import torch
import tqdm
from torch import optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

from load_dataset import make_datasets, make_pairs
from load_osu_map import load_osu_maps
from make_osu_file import tensor_to_osu_map
from models import OsuModel
import torch

def guided_attn_loss(attn, g=0.2):
    B, T_dec, T_enc = attn.size()
    W = torch.arange(T_dec, device=attn.device).float().unsqueeze(1) / T_dec
    J = torch.arange(T_enc, device=attn.device).float().unsqueeze(0) / T_enc
    G = 1.0 - torch.exp(-(W - J) ** 2 / (2 * g * g))
    loss = (attn * G.unsqueeze(0)).mean()
    return loss

def flatten_for_ce(logits, targets):
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1).long()
    return logits, targets

def positional_loss(out, real, sigma=0.1):
    B, T, D = out.shape
    device = out.device
    pred_probs = F.softmax(out, dim=-1)  # [B, T, D]

    positions = torch.arange(D, device=device).float().view(1, 1, D)
    real = real.unsqueeze(-1).float()
    gauss = torch.exp(-0.5 * ((positions - real) / sigma) ** 2)
    target_probs = gauss / gauss.sum(dim=-1, keepdim=True)  # [B, T, D]
    pred_cdf = torch.cumsum(pred_probs, dim=-1)
    target_cdf = torch.cumsum(target_probs, dim=-1)

    wdist = torch.abs(pred_cdf - target_cdf).mean()
    return wdist

lr = 1e-3

MEL_DIM = 80
epochs = 4000

data_list = load_osu_maps()
pairs = make_pairs(data_list, 1)

dataset, dataloader = make_datasets(pairs=pairs, batch_size=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = OsuModel(MEL_DIM, 250, 11, 6, num_layer=3)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

ce_loss = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
l1_loss = nn.L1Loss()
if os.path.exists("t2.pt"):
    model.load_state_dict(torch.load("t2.pt"))

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for batch in tqdm.tqdm(dataloader):
        mel, audio_mask, seq, seq_mask, meta = batch

        mel = mel.to(device)
        meta = meta.to(device)
        seq = seq.to(device)
        seq_mask = seq_mask.to(device)
        audio_mask = audio_mask.to(device)
        steps = seq.size(1)

        out_x, out_y, out_dt, out_type, out_stop, alignment = model(
            inputs=mel,
            truth=seq[:, :-1],
            pad_mask=audio_mask,
            meta=meta,
            steps=None,
            teacher_forcing_ratio=1.,
            )

        truth_x = seq[:, 1:, 0] / 1000
        truth_y = seq[:, 1:, 1] / 1000
        truth_dt = seq[:, 1:, 2].float()
        truth_type = seq[:, 1:, 3]

        stops = seq_mask

        loss_x = l1_loss(out_x, truth_x)
        loss_y = l1_loss(out_y, truth_y)

        logits_type, targets_type = flatten_for_ce(out_type, truth_type)
        loss_type = ce_loss(logits_type, targets_type)

        loss_stop = bce_loss(out_stop, stops[:, 1:].float())

        loss_dt = mse_loss(out_dt, truth_dt.float() / 1000.0)

        loss_map = loss_x + loss_y + loss_type + loss_dt + loss_stop
        attn_loss = guided_attn_loss(alignment, g=0.1)

        loss = loss_map + attn_loss
        if epoch % 100 == 0:
            plt.imshow(alignment[0].detach().cpu().numpy())
            plt.show()

        if epoch % 20 == 0:
            torch.save(model.state_dict(), "t2.pt")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - LR: {scheduler.get_lr()[0]}")

print("Training complete.")