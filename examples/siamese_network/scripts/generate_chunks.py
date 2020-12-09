import concurrent.futures
import datetime
import glob
import os
import progressbar
import random
import re
import srt
import torchaudio


class GenerateChunks:
    dataset_path = None
    dataset_wav_full_path = None
    dataset_srt_full_path = None
    dataset_wav_chunks_path = None
    dataset_txt_chunks_path = None
    srt_files = None

    def __init__(self, dataset_name):
        root_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data'
        )
        self.dataset_path = os.path.join(root_path, dataset_name)
        self.dataset_wav_full_path = os.path.join(
            self.dataset_path, 'wav_full'
        )
        self.dataset_srt_full_path = os.path.join(
            self.dataset_path, 'srt_full'
        )
        self.dataset_wav_chunks_path = os.path.join(self.dataset_path, 'wav')
        self.dataset_txt_chunks_path = os.path.join(self.dataset_path, 'txt')

    def run(self):
        # Request user input: maximum number of threads to run
        max_threads = None
        while max_threads is None or \
                not max_threads.isdigit() or int(max_threads) < 1:
            max_threads = input('How many threads do you want to run? ')
        max_threads = int(max_threads)

        # Initialize pipeline
        self._initialize()

        # Use a ThreadPoolExecutor to process samples in parallel
        bar = progressbar.ProgressBar(max_value=len(self.srt_files))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) \
                as executor:
            for i, srt_file_path in enumerate(self.srt_files):
                executor.submit(self._process_item, srt_file_path, bar, i)

    def _initialize(self):
        # Create folders to store the chunks
        if not os.path.exists(self.dataset_wav_chunks_path):
            os.makedirs(self.dataset_wav_chunks_path)
        if not os.path.exists(self.dataset_txt_chunks_path):
            os.makedirs(self.dataset_txt_chunks_path)

        # Get a list of the available SRT files
        self.srt_files = glob.glob(
            os.path.join(self.dataset_srt_full_path, '*.srt')
        )

        # Verify that the list of srt files is not empty
        if len(self.srt_files) < 1:
            exit('No files have been found')

    def _process_item(self, srt_file_path, bar, i):
        # Read the downloaded SRT file and obtain its ID
        srt_file_id = re.sub(
            r'\.[\w]*\.srt', '', os.path.basename(srt_file_path)
        )
        srt_file = GenerateChunks.clean_srt_file(
            srt.parse(open(srt_file_path, 'r').read())
        )

        # Read the WAV file from disk
        wav_file_path = os.path.join(
            self.dataset_wav_full_path, srt_file_id + '.wav'
        )
        wav, sf = torchaudio.load(wav_file_path)

        # Iterate over each SRT line
        for srt_item in srt_file:
            self.save_chunk(wav, srt_item, sf)

        # Update the ProgressBar
        bar.update(max(bar.value, i))

    @staticmethod
    def clean_srt_file(srt_file):
        srt_file_cleaned = []
        previous_line_content = None
        for srt_file_line in srt_file:
            srt_file_line_length = srt_file_line.end - srt_file_line.start
            srt_file_line_length_seconds = \
                srt_file_line_length.seconds * 10E5 + \
                srt_file_line_length.microseconds
            if srt_file_line_length_seconds > 10000:
                line_parts = [
                    x.rstrip() for x in srt_file_line.content.split('\n')
                ]
                # Automatic format from YouTube always has 2 lines
                if len(line_parts) > 1:
                    line_parts.remove('') if '' in line_parts else \
                        line_parts.remove(previous_line_content)
                srt_file_line.content = line_parts[0].rstrip()
                previous_line_content = line_parts[0].rstrip()
                srt_file_cleaned.append(srt_file_line)
        return list(srt_file_cleaned)

    def save_chunk(self, wav, srt_line, sf):
        start_sample, end_sample = GenerateChunks.compute_cut_limits(
            srt_line.start, srt_line.end, sf
        )
        chunk_name = None
        while chunk_name is None or os.path.exists(
                os.path.join(self.dataset_wav_chunks_path, chunk_name)
        ):
            chunk_name = ''.join(
                [str(random.randint(0, 9)) for _ in range(10)]
            ) + '.wav'
        torchaudio.save(
            os.path.join(self.dataset_wav_chunks_path, chunk_name),
            wav[:, start_sample:end_sample],
            sf
        )
        open(os.path.join(
            self.dataset_txt_chunks_path,
            chunk_name.replace('wav', 'txt')
        ), 'wt').write(srt_line.content)

    @staticmethod
    def compute_cut_limits(start_time, end_time, sf):
        start_time_float = (
                start_time.seconds + start_time.microseconds / 10E6
        ) if isinstance(start_time, datetime.timedelta) else float(start_time)
        end_time_float = (
                end_time.seconds + end_time.microseconds / 10E5
        ) if isinstance(end_time, datetime.timedelta) else float(end_time)
        return int(start_time_float * sf), int(end_time_float * sf)
