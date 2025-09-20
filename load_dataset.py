import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from load_osu_map import parse_one_osu_map, load_osu_maps, Slider, ExtraType, Circle

class MyDataset(Dataset):
    def __init__(self, pairs, max_len=2000, sr=44100, n_mels=80, hop_length=512):
        self.pairs = pairs
        self.max_len = max_len
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        audio_path, meta, obj_list = self.pairs[idx]

        wav, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=self.sr,
            n_fft=1024,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=30,
            fmax=self.sr // 2
        )
        mel = librosa.power_to_db(mel, ref=1.0)
        mel = np.clip((mel + 100) / 100, 0, 1)
        mel = torch.from_numpy(mel.T).float()

        if mel.size(0) > self.max_len:
            mel = mel[:self.max_len]
        else:
            pad_len = self.max_len - mel.size(0)
            mel = torch.cat([mel, torch.zeros(pad_len, self.n_mels)], dim=0)

        seq = torch.tensor(obj_list, dtype=torch.long)
        seq[1:, 2] = seq[1:, 2] - seq[:-1, 2]
        return mel, meta, seq


def collate_fn(batch):
    """
    batch: list of tuples (audio, meta, seq)
      - audio: [T_audio, feat_dim] (variable length)
      - meta: [meta_dim]
      - seq: [T_seq, 4] (x, y, dt, type)
    """
    audios, metas, seqs = zip(*batch)
    metas = [torch.tensor(list(map(float, meta.values()))) for meta in metas]
    # === Audio padding ===
    audio_lengths = [a.size(0) for a in audios]
    max_audio_len = max(audio_lengths)
    feat_dim = audios[0].size(1)

    padded_audios = []
    audio_masks = []
    for a in audios:
        pad_len = max_audio_len - a.size(0)
        padded = torch.cat([a, torch.zeros(pad_len, feat_dim)], dim=0)
        padded_audios.append(padded)
        audio_masks.append(torch.cat([torch.zeros(a.size(0)), torch.ones(pad_len)]))

    padded_audios = torch.stack(padded_audios, dim=0)          # [B, max_audio_len, feat_dim]
    audio_masks = torch.stack(audio_masks, dim=0).bool()       # [B, max_audio_len]

    seq_lengths = [s.size(0) for s in seqs]
    max_seq_len = max(seq_lengths)

    padded_seqs = []
    seq_masks = []
    for s in seqs:
        pad_len = max_seq_len - s.size(0)
        pad = torch.cat([torch.zeros(pad_len, 3, dtype=torch.long).fill_(0), torch.zeros(pad_len, 1, dtype=torch.long).fill_(10)], dim=1)
        padded = torch.cat([s, pad], dim=0)
        padded_seqs.append(padded)
        seq_masks.append(torch.cat([torch.zeros(s.size(0)), torch.ones(pad_len)]))

    padded_seqs = torch.stack(padded_seqs, dim=0)
    seq_masks = torch.stack(seq_masks, dim=0).bool()

    metas = torch.stack(metas, dim=0)

    return padded_audios, audio_masks, padded_seqs, seq_masks, metas


def get_hit_normal_info(hit_obj):
    return [hit_obj.x, hit_obj.y, hit_obj.t, hit_obj.type]

def make_pairs(data_list, num_data):
    num_map = 0
    pairs = []
    for audio_path, text_path in data_list:
        info = parse_one_osu_map(text_path)
        if info["General"]["Mode"] != "0":
            continue
        meta = {k: float(v) for k, v in info["Difficulty"].items()}
        hit_obj = info["HitObjects"]["nodes"]
        out_obj_list = []
        for hit_obj in hit_obj:
            if isinstance(hit_obj, Slider):
                out_obj_list.append(get_hit_normal_info(hit_obj))

                for i in range(hit_obj.slides):
                    all_slides = []
                    slider_type, points = hit_obj.param.curve_type, hit_obj.param.curvePoints
                    for point in points:
                        x, y = map(int, point.split(":"))
                        x = min(max(x, 0), 1000)
                        y = min(max(y, 0), 1000)
                        out_obj_list.append([x, y, hit_obj.t, ExtraType.slider_body])
                    if i % 2 == 1:
                        all_slides.append([hit_obj.x, hit_obj.y, hit_obj.t, ExtraType.slider_slide])
                        all_slides = all_slides[::-1]
                    all_slides.append([hit_obj.x, hit_obj.y, hit_obj.t, ExtraType.slider_end])

                    out_obj_list.extend(all_slides)
            elif isinstance(hit_obj, Circle):
                out_obj_list.append(get_hit_normal_info(hit_obj))
        num_map += 1
        out_obj_list.append([0, 0, 0, 0])
        pairs.append([audio_path, meta, out_obj_list])

        if num_map >= num_data:
            break
    return pairs

def make_datasets(pairs, batch_size=8, shuffle=True):
    dataset = MyDataset(pairs)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=collate_fn)
    return dataset, dataloader
