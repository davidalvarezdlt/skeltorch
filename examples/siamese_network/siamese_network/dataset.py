import os.path
import random
import torchaudio
import torch
import torch.utils.data
import torch.nn.functional as F


class SiameseDataset(torch.utils.data.Dataset):
    data_path = None
    data_meta = None
    input_samples = None
    max_samples = None
    pad_short = None
    transform_resample = None
    transform_mel = None

    def __init__(self, data_path, data_meta, sf_original, sf_target,
                 cut_length, n_mfcc, n_fft, win_length, hop_length,
                 max_samples, pad_short=False):
        self.data_path = data_path
        self.data_meta = data_meta
        self.input_samples = sf_target * cut_length
        self.max_samples = max_samples
        self.pad_short = pad_short
        self.transform_resample = torchaudio.transforms.Resample(
            sf_original, sf_target
        )
        self.transform_mel = torchaudio.transforms.MFCC(
            sf_target, n_mfcc=n_mfcc, melkwargs={
                'n_fft': n_fft, 'win_length': win_length,
                'hop_length': hop_length
            }
        )

    def __getitem__(self, index):
        speakers_keys = random.sample(self.data_meta.keys(), 2)
        if random.randint(0, 1) == 1:
            speakers_keys[1] = speakers_keys[0]

        chunk_1 = self._load_chunk(
            random.sample(self.data_meta[speakers_keys[0]], 1)[0]
        )
        chunk_2 = self._load_chunk(
            random.sample(self.data_meta[speakers_keys[1]], 1)[0]
        )
        is_different_speaker = 0 if speakers_keys[0] == speakers_keys[1] else 1
        return chunk_1, chunk_2, is_different_speaker

    def _load_chunk(self, utterance_path):
        chunk_data, _ = torchaudio.load(
            os.path.join(self.data_path, utterance_path)
        )
        chunk_data = self.transform_resample(chunk_data)
        if chunk_data.size(0) > 1:
            chunk_data = torch.mean(chunk_data, dim=0, keepdim=True)
        if self.pad_short and chunk_data.size(1) < self.input_samples:
            chunk_pad = self.input_samples - chunk_data.size(1)
            chunk_data = F.pad(chunk_data, (0, chunk_pad))
        chunk_start = random.randint(
            0, chunk_data.size(1) - self.input_samples
        )
        return self.transform_mel(
            chunk_data[:, chunk_start:chunk_start + self.input_samples]
        )

    def __len__(self):
        return self.max_samples
