import librosa.effects
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
                'n_fft': n_fft,
                'win_length': win_length,
                'hop_length': hop_length
            }
        )

    def __getitem__(self, index):
        # Random choice of speakers, where probability of being the same
        # is p=0.5
        speakers_keys = random.sample(self.data_meta.keys(), 2)
        if random.randint(0, 1) == 1:
            speakers_keys[1] = speakers_keys[0]

        # Get random data selecting random samples from the data set
        chunks_paths = [
            random.sample(self.data_meta[speaker_id], 1)[0]
            for speaker_id in speakers_keys
        ]
        chunks_data = [
            self.get_chunk_data(utterance_path)
            for utterance_path in chunks_paths
        ]

        # Return (first_utterance_data, second_utterance_data,
        # is_different_speaker)
        return chunks_data[0], chunks_data[1], \
            0 if speakers_keys[0] == speakers_keys[1] else 1

    def __len__(self):
        return self.max_samples

    def get_chunk_data(self, utterance_path):
        chunk_data, _ = torchaudio.load(
            os.path.join(self.data_path, utterance_path)
        )
        chunk_data = self.transform_resample(chunk_data)
        chunk_data, _ = librosa.effects.trim(chunk_data)
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

    def get_random_samples(self, speaker_id, n_random_samples):
        chunks_paths = [
            random.sample(self.data_meta[speaker_id], 1)[0]
            for _ in range(n_random_samples)
        ]
        chunks_data = [
            self.get_chunk_data(chunk_path)
            for chunk_path in chunks_paths
        ]
        return torch.stack(chunks_data, dim=0)
