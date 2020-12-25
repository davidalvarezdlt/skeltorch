from .dataset import SiameseDataset
import librosa.effects
import os.path
import progressbar
import random
import skeltorch
import torch.utils.data
import torchaudio


class SiameseData(skeltorch.Data):
    data_meta_train = {}
    data_meta_validation = {}
    data_meta_test = {}

    def create(self, data_path):
        # Download VCTK dataset
        torchaudio.datasets.VCTK(root=data_path, download=True)

        # Get list of files
        vctk_folder = os.path.join(
            data_path, torchaudio.datasets.vctk.FOLDER_IN_ARCHIVE
        )
        vctk_files_list = list(
            torchaudio.datasets.utils.walk_files(vctk_folder, suffix='.wav')
        )

        # Fill items with partial path
        for i, vctk_item in enumerate(vctk_files_list):
            speaker_id, utterance_id = vctk_item.split('_')
            vctk_files_list[i] = os.path.join(
                torchaudio.datasets.vctk.FOLDER_IN_ARCHIVE,
                'wav48',
                speaker_id,
                vctk_item
            )

        # Remove wrong samples
        vctk_files_list = self._clean_wrong_audios(data_path, vctk_files_list)

        # Create a dictionary containing samples for each speaker
        vctk_files_dict = {}
        for vctk_file_item in vctk_files_list:
            speaker_id = vctk_file_item[-12:-8]
            if speaker_id not in vctk_files_dict:
                vctk_files_dict[speaker_id] = [vctk_file_item]
            else:
                vctk_files_dict[speaker_id].append(vctk_file_item)

        # Store test samples and remove the speakers from the dict
        n_test_speakers = self.get_conf(
            'data', 'n_test_speakers'
        )
        test_speakers = random.sample(vctk_files_dict.keys(), n_test_speakers)
        for speaker_id in test_speakers:
            self.data_meta_test[speaker_id] = vctk_files_dict[speaker_id]
            vctk_files_dict.pop(speaker_id)

        # Create train/validation split for the rest of speakers
        val_split = self.get_conf('data', 'val_split')
        for speaker_id in vctk_files_dict.keys():
            n_validation_samples = round(
                len(vctk_files_dict[speaker_id]) * val_split
            )
            validation_samples = random.sample(
                vctk_files_dict[speaker_id], n_validation_samples
            )
            self.data_meta_validation[speaker_id] = validation_samples
            self.data_meta_train[speaker_id] = list(
                set(vctk_files_dict[speaker_id]) - set(validation_samples)
            )

    def _clean_wrong_audios(self, data_path, vctk_files_list):
        self.logger.info('Validating audio files...')
        bar = progressbar.ProgressBar(max_value=len(vctk_files_list))
        min_samples = self.get_conf('data', 'sf_original') * \
            self.get_conf('data', 'cut_length')
        wrong_audios = set()
        for i, vctk_item in enumerate(vctk_files_list):
            try:
                item_d, _ = torchaudio.load(os.path.join(data_path, vctk_item))
                item_d, _ = librosa.effects.trim(item_d)
                if item_d.size(1) < min_samples:
                    wrong_audios.add(vctk_item)
            except FileNotFoundError:
                wrong_audios.add(vctk_item)
            bar.update(i)
        return list(set(vctk_files_list) - wrong_audios)

    def load_datasets(self, data_path):
        cut_length = self.get_conf('data', 'cut_length')
        self.datasets['train'] = SiameseDataset(
            data_path=data_path,
            data_meta=self.data_meta_train,
            sf_original=self.get_conf('data', 'sf_original'),
            sf_target=self.get_conf('data', 'sf_target'),
            cut_length=cut_length,
            n_mfcc=self.get_conf('data', 'n_mfcc'),
            n_fft=self.get_conf('data', 'n_fft'),
            win_length=self.get_conf('data', 'win_length'),
            hop_length=self.get_conf('data', 'hop_length'),
            max_samples=self.get_conf('training', 'train_max_samples')
        )
        self.datasets['validation'] = SiameseDataset(
            data_path=data_path,
            data_meta=self.data_meta_validation,
            sf_original=self.get_conf('data', 'sf_original'),
            sf_target=self.get_conf('data', 'sf_target'),
            cut_length=cut_length,
            n_mfcc=self.get_conf('data', 'n_mfcc'),
            n_fft=self.get_conf('data', 'n_fft'),
            win_length=self.get_conf('data', 'win_length'),
            hop_length=self.get_conf('data', 'hop_length'),
            max_samples=self.get_conf('training', 'validation_max_samples')
        )
        self.datasets['test'] = SiameseDataset(
            data_path=data_path,
            data_meta=self.data_meta_test,
            sf_original=self.get_conf('data', 'sf_original'),
            sf_target=self.get_conf('data', 'sf_target'),
            cut_length=cut_length,
            n_mfcc=self.get_conf('data', 'n_mfcc'),
            n_fft=self.get_conf('data', 'n_fft'),
            win_length=self.get_conf('data', 'win_length'),
            hop_length=self.get_conf('data', 'hop_length'),
            max_samples=self.get_conf('training', 'test_max_samples')
        )

    def load_loaders(self, data_path, num_workers):
        self.loaders['train'] = torch.utils.data.DataLoader(
            dataset=self.datasets['train'],
            shuffle=True,
            batch_size=self.get_conf('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['validation'] = torch.utils.data.DataLoader(
            dataset=self.datasets['validation'],
            shuffle=True,
            batch_size=self.get_conf('training', 'batch_size'),
            num_workers=num_workers
        )
        self.loaders['test'] = torch.utils.data.DataLoader(
            dataset=self.datasets['test'],
            shuffle=True,
            batch_size=self.get_conf('training', 'batch_size'),
            num_workers=num_workers
        )
