import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import progressbar
import pydub
import pydub.playback
import random
import skeltorch
import siamese_network.dataset
import siamese_network.model
import siamese_network.runner
import torch.utils.data
import torch.nn.functional as F


class CleanSpeakers:
    data_path = None
    experiments_path = None
    dataset_wav_chunks_path = None
    dataset_txt_chunks_path = None
    dataset_test_set_path = None
    wav_chunks_list = None
    txt_chunks_list = None
    speaker_chunks_indexes = None
    non_speaker_chunks_indexes = None

    def __init__(self, dataset_name):
        self.data_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data'
        )
        self.experiments_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'experiments'
        )
        self.dataset_wav_chunks_path = os.path.join(
            self.data_path, dataset_name, 'wav'
        )
        self.dataset_txt_chunks_path = os.path.join(
            self.data_path, dataset_name, 'txt'
        )
        self.dataset_test_set_path = os.path.join(
            self.data_path, dataset_name, 'test_set.pkl'
        )

    def run(self):
        # Initialize pipeline
        self._initialize()

        # Annotate a test split to use as the input of the Siamese network
        if not os.path.exists(self.dataset_test_set_path):
            self._create_test_split()
        else:
            self.speaker_chunks_indexes, self.non_speaker_chunks_indexes = \
                pickle.load(open(self.dataset_test_set_path, 'rb'))

        # Request user input: device
        device = None
        while device is None or device not in ['cpu', 'cuda']:
            device = input(
                'Which device do you want to use to run the model? [cpu/cuda] '
            )

        # Request user input: checkpoint path
        experiment_name = None
        while experiment_name is None or not os.path.exists(
                os.path.join(self.experiments_path, experiment_name)
        ):
            experiment_name = input(
                'What experiment do you want to use as Siamese network? (make '
                'sure it exists) '
            )

        # Request user input: checkpoint number
        checkpoint_path = None
        while checkpoint_path is None or not os.path.exists(checkpoint_path):
            checkpoint_number = input(
                'What checkpoint do you want to use? (make sure it exists) '
            )
            checkpoint_path = os.path.join(
                self.experiments_path, experiment_name,
                'checkpoints', '{}.checkpoint.pkl'.format(checkpoint_number)
            )

        # Load configuration of the experiment
        configuration = skeltorch.Configuration(None)
        configuration.load(
            os.path.join(self.experiments_path, experiment_name, 'config.pkl')
        )

        # Create and load the Siamese network
        checkpoint_data = torch.load(
            open(checkpoint_path, 'rb'), map_location=device
        )
        model = siamese_network.model.SiameseNetwork(
            n_mfcc=configuration.get('data', 'n_mfcc'),
            sf=configuration.get('data', 'sf_target'),
            cut_length=configuration.get('data', 'cut_length'),
            hop_length=configuration.get('data', 'hop_length'),
            n_components=configuration.get('model', 'n_components')
        ).to(device)
        model.load_state_dict(checkpoint_data['model'])

        # Create data set with annotated data to draw threshold curves.
        # Set n^2 input sample pairs, where n is the number of annotated
        # samples
        chunks_annotated = {
            'speaker': [
                self.wav_chunks_list[item_index]
                for item_index in self.speaker_chunks_indexes
            ],
            'non_speaker': [
                self.wav_chunks_list[item_index]
                for item_index in self.non_speaker_chunks_indexes
            ]
        }
        chunks_test_dataset = siamese_network.dataset.SiameseDataset(
            data_path=self.data_path,
            data_meta=chunks_annotated,
            sf_original=configuration.get('data', 'sf_original'),
            sf_target=configuration.get('data', 'sf_target'),
            cut_length=configuration.get('data', 'cut_length'),
            n_mfcc=configuration.get('data', 'n_mfcc'),
            n_fft=configuration.get('data', 'n_fft'),
            win_length=configuration.get('data', 'win_length'),
            hop_length=configuration.get('data', 'hop_length'),
            max_samples=len(self.speaker_chunks_indexes) ** 2,
            pad_short=True
        )

        # Obtain both GT data and predicted output
        gt, pred = [], []
        for it_data in torch.utils.data.DataLoader(
                chunks_test_dataset, shuffle=True, batch_size=32
        ):
            with torch.no_grad():
                y1, y2 = model(it_data[0].to(device), it_data[1].to(device))
            gt += it_data[2].tolist()
            pred += (F.pairwise_distance(y1, y2)).tolist()

        # Compute metrics of the predictions
        tp, fp, tn, fn, precision, recall, f1 = \
            siamese_network.runner.SiameseRunner.compute_metrics(
                np.array(gt), np.array(pred), 4, 1000
            )

        # Draw Precision(Recall) curve
        plt.figure(1)
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim(left=0, right=1)
        plt.ylim(bottom=0, top=1)
        plt.show()

        # Draw Precision(Threshold) curve
        plt.figure(2)
        plt.plot(np.linspace(0, 4, 1000), precision)
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.xlim(left=0, right=4)
        plt.ylim(bottom=0, top=1)
        plt.show()

        # Draw Recall(Threshold) curve
        plt.figure(3)
        plt.plot(np.linspace(0, 4, 1000), recall)
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.xlim(left=0, right=4)
        plt.ylim(bottom=0, top=1)
        plt.show()

        # Draw F-Score(Threshold) curve
        plt.figure(4)
        plt.plot(np.linspace(0, 4, 1000), f1)
        plt.xlabel('Threshold')
        plt.ylabel('F-Score')
        plt.xlim(left=0, right=4)
        plt.ylim(bottom=0, top=1)
        plt.show()

        # Request user input: threshold value
        threshold = None
        while threshold is None \
                or not threshold.replace('.', '', 1).isdigit() \
                or not float(threshold) > 0:
            threshold = input('Which threshold do you want to set? ')
        threshold = float(threshold)

        # Request user input: number of comparisons per sample
        n_comparisons = None
        while n_comparisons is None \
                or not n_comparisons.isdigit() \
                or int(n_comparisons) < 1:
            n_comparisons = input(
                'How many comparisions per chunk do you want to perform? '
            )
        n_comparisons = int(n_comparisons)

        # Clean the chunks using n_comparisons per chunk. Set this number as
        # batch size for simplicity.
        bar = progressbar.ProgressBar(max_value=len(self.wav_chunks_list))
        for i, chunk_item_path in enumerate(self.wav_chunks_list):
            if i in self.speaker_chunks_indexes \
                    or i in self.non_speaker_chunks_indexes:
                continue

            # Create data associated to the current chunk to analyze
            x1 = torch.stack([
                chunks_test_dataset.get_chunk_data(chunk_item_path)
                for _ in range(n_comparisons)
            ])

            # Create data associated to GT speaker
            x2 = chunks_test_dataset.get_random_samples(
                'speaker', n_comparisons
            )

            # Propagate data through the model
            with torch.no_grad():
                y1, y2 = model(x1.to(device), x2.to(device))

            # Measure distance and determine if the samples are from the same
            # speaker
            chunks_distance = F.pairwise_distance(y1, y2)
            different_speaker_count = (chunks_distance > threshold).sum()

            # Check if the number of counts is >= n_comparisons + 1 // 2. If
            # so, remove samples from disk
            if different_speaker_count >= n_comparisons / 2:
                os.remove(chunk_item_path)
                os.remove(self.txt_chunks_list[i])

            # Update bar
            bar.update(i)

    def _initialize(self):
        self.wav_chunks_list = sorted(
            glob.glob(os.path.join(self.dataset_wav_chunks_path, '*.wav'))
        )
        self.txt_chunks_list = sorted(
            glob.glob(os.path.join(self.dataset_txt_chunks_path, '*.txt'))
        )
        if len(self.wav_chunks_list) != len(self.txt_chunks_list):
            exit('Different number of audios and transcriptions.')

    def _create_test_split(self, min_samples=10):
        # Request user input: number of test samples to generate
        n_chunks = None
        while n_chunks is None \
                or not n_chunks.isdigit() or int(n_chunks) < min_samples:
            n_chunks = input(
                'How many test samples do you want to annotate (min. {})? '
                .format(min_samples)
            )
        n_chunks = int(n_chunks)

        # Initialize a list to store the chunks to use as test set
        self.speaker_chunks_indexes = []
        self.non_speaker_chunks_indexes = []

        # Print initial separator
        print('=' * 50 + '\n')

        # Iterate while not enough samples have been stored
        while len(self.speaker_chunks_indexes) < n_chunks \
                or len(self.non_speaker_chunks_indexes) < n_chunks:

            # Take a random sample
            random_index = random.randint(0, len(self.wav_chunks_list) - 1)

            # Verify that the random sample is not in any of the lists
            if random_index in self.speaker_chunks_indexes \
                    or random_index in self.non_speaker_chunks_indexes:
                continue

            # Load WAV with PyDub
            wav = pydub.AudioSegment.from_wav(
                self.wav_chunks_list[random_index]
            )
            txt = open(self.txt_chunks_list[random_index], 'rt').read()

            # Print the text of the chunk and play the chunk
            print('Speaker chunks in the list: {}/{}'.format(
                len(self.speaker_chunks_indexes), n_chunks)
            )
            print('Non-speaker chunks in the list: {}/{}'.format(
                len(self.non_speaker_chunks_indexes), n_chunks)
            )
            print('Chunk Text: {}\n'.format(txt))
            pydub.playback.play(wav)

            # Request the user whether he wants to include or not the chunk to
            # the test set
            user_input = None
            while user_input not in ['y', 'n']:
                user_input = input(
                    'Is the chunk from your desired speaker? [y/n]: '
                )
                user_input = user_input.lower()

            # Print final separator
            print('\n' + '=' * 50 + '\n')

            # If the user has request to include the chunk in the list and
            # there is space, add it
            if user_input == 'y' \
                    and len(self.speaker_chunks_indexes) < n_chunks:
                self.speaker_chunks_indexes.append(random_index)

            # If the answer is negative
            if user_input == 'n' \
                    and len(self.non_speaker_chunks_indexes) < n_chunks:
                self.non_speaker_chunks_indexes.append(random_index)

        # Save the list inside self.dataset_test_set_path
        with open(self.dataset_test_set_path, 'wb') as test_set_file:
            pickle.dump(
                (self.speaker_chunks_indexes, self.non_speaker_chunks_indexes),
                test_set_file
            )
