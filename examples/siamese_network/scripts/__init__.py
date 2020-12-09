from .clean_speakers import CleanSpeakers
from .create_list import CreateList
from .download_data import DownloadData
from .generate_chunks import GenerateChunks


class SpeechDatasetGenerator:
    dataset_name = None
    create_list_handler = None
    download_data_handler = None
    generate_chunks_handler = None
    clean_speakers_handler = None
    command_handlers = None

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.create_list_handler = CreateList(self.dataset_name)
        self.download_data_handler = DownloadData(self.dataset_name)
        self.generate_chunks_handler = GenerateChunks(self.dataset_name)
        self.clean_speakers_handler = CleanSpeakers(self.dataset_name)
        self.command_handlers = {
            'create_list': self.create_list_handler.run,
            'download_data': self.download_data_handler.run,
            'generate_chunks': self.generate_chunks_handler.run,
            'clean_speakers': self.clean_speakers_handler.run
        }

    def run(self, command):
        self.command_handlers[command]()
