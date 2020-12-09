from . import SpeechDatasetGenerator
import argparse

# Create main parser and subparsers
parser = argparse.ArgumentParser(description='Run Speech Dataset Generator')
parser.add_argument(
    '--dataset-name', required=True, help='Name of the dataset to create'
)

# Create subparsers associated to each command
subparsers = parser.add_subparsers(dest='command')
subparser_generate_list = subparsers.add_parser(name='create_list')
subparser_download_data = subparsers.add_parser(name='download_data')
subparser_generate_chunks = subparsers.add_parser(name='generate_chunks')
subparser_generate_test_set = subparsers.add_parser(name='generate_test_set')
subparser_clean_speakers = subparsers.add_parser(name='clean_speakers')

# Parse arguments and run appropriate command
args = parser.parse_args()
SpeechDatasetGenerator(args.dataset_name).run(args.command)
