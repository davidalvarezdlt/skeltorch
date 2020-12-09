import os
import glob
import shutil


class DownloadData:
    dataset_path = None
    dataset_list_path = None
    dataset_wav_files_path = None
    dataset_srt_files_path = None

    def __init__(self, dataset_name):
        root_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data'
        )
        self.dataset_path = os.path.join(root_path, dataset_name)
        self.dataset_list_path = os.path.join(self.dataset_path, 'list.txt')
        self.dataset_wav_files_path = os.path.join(
            self.dataset_path, 'wav_full'
        )
        self.dataset_srt_files_path = os.path.join(
            self.dataset_path, 'srt_full'
        )

    def run(self):
        # Request user input: language to use
        lang = None
        while lang is None or lang == '':
            lang = input(
                'What is the ISO 639-1 code of the language of the '
                'transcriptions? '
            )
        lang = lang.lower()

        # Initialize pipeline
        self._initialize()

        # Run youtube-dl command
        os.system(
            'youtube-dl --no-check-certificate -a {} -o {} -x --audio-format '
            'wav --write-auto-sub --sub-lang {} '
            '--convert-subs srt'.format(
                self.dataset_list_path, os.path.join(
                    self.dataset_wav_files_path, r'%\(id\)s.%\(ext\)s'
                ), lang
            )
        )

        # Move .srt files to their correct folder and remove items without
        # transcription
        for srt_file in glob.glob(
                os.path.join(self.dataset_wav_files_path, '*.srt')
        ):
            shutil.move(srt_file, self.dataset_srt_files_path)
        self._remove_items_without_transcription(lang)

    def _initialize(self):
        if not os.path.exists(self.dataset_list_path):
            raise FileNotFoundError(
                'List file containing the videos to download not found.'
            )
        if not os.path.exists(self.dataset_wav_files_path):
            os.makedirs(self.dataset_wav_files_path)
        if not os.path.exists(self.dataset_srt_files_path):
            os.makedirs(self.dataset_srt_files_path)

    def _remove_items_without_transcription(self, lang):
        for wav_file in glob.glob(
                os.path.join(self.dataset_wav_files_path, '*.wav')
        ):
            wav_file_id = os.path.splitext(os.path.basename(wav_file))[0]
            srt_file_path = os.path.join(
                self.dataset_srt_files_path,
                wav_file_id + '.{}.srt'.format(lang)
            )
            if not os.path.exists(srt_file_path):
                os.remove(wav_file)
