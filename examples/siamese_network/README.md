# Speech Dataset Generator
The goal of this repository is to provide a pipeline to generate speech data sets using YouTube videos as a source. It
may be useful especially for text-to-speech systems where we want to synthesize the voice of famous people. The pipeline
is composed of various steps:

1. Create a list of YouTube videos related to a certain keyword using
[YouTube Data API](https://developers.google.com/youtube/v3).
2. Download the audios and their (aligned) transcriptions using `youtube-dl`.
3. Use aligned transcriptions to create a set of audio chunks.
4. Remove those chunks that don't belong to the voice we want to retrieve.

Before starting, remember to install the dependencies with:

```
pip install -r requirements.txt
```

## 1. Creating a list of relevant videos
The first step is to create a list of videos related to a certain person. To do so, we will use the official
YouTube Data API. We will need to
[obtain an API key](https://developers.google.com/youtube/registering_an_application?hl=en) inside the Google Cloud
Platform.

```
Usage:
    python -m speech_dataset_generator --dataset-name <experiment_name> create_list

Example:
    $ python -m speech_dataset_generator --dataset-name donald_dataset create_list
    $ What keyword do you want to use to find related videos? donald trump speech
    $ How many videos do you want to list? 200
    $ Introduce your YouTube Data API Key? AIzezyC13omae4wAzW86wuOmTIsxbuzZucwsapA
```

The script will create the folder `data/<dataset_name>` containing a file named `list.txt`.

## 2. Downloading audios and transcriptions
With our list of related videos to use, the next step is to download the audios and their transcriptions. To do so, we
will use [youtube-dl](https://github.com/ytdl-org/youtube-dl). Make sure to have it installed along with a tool to
extract the audios (for instance, `ffmpeg`).

```
Usage:
    python -m speech_dataset_generator --dataset-name <experiment_name> download_data

Example:
    $ python -m speech_dataset_generator --dataset-name donald_dataset download_data
    $ What is the ISO 639-1 code of the language of the transcriptions? en
```

Downloaded data will be saved inside `data/<dataset_name>/wav_full` and `data/<dataset_name>/srt_full`. Notice that the
audios without a transcription will be automatically removed.

## 3. Splitting audios into chunks
The next step is to use the implicit alignment of the transcriptions to cut the audios and create small chunks. As the
alignment provided by YouTube is far from perfect, better results would be obtained with additional processing steps.

```
Usage:
    python -m speech_dataset_generator --dataset-name <experiment_name> generate_chunks

Example:
    $ python -m speech_dataset_generator --dataset-name donald_dataset generate_chunks
    $ How many threads do you want to run? 10
```

Chunks and their transcription will be stored in `data/<dataset_name>/wav` and `data/<dataset_name>/txt`, respectively.

**Important: some videos have manual transcriptions, which may not be correct. For instance, it may be the case that the
publisher of the video uses captions to narrate the scene. To solve that problem, there should be another processing
layer that verifies that the transcription is correct.**

## 4. Cleaning chunks from wrong speakers
Finally, the last step is to remove those chunks which don't belong to our target speaker. To do so, we provide you with
a simple Siamese network inside `siamese_network/`. The package uses
[Skeltorch](https://github.com/davidalvarezdlt/skeltorch), which makes it very easy to share and create experiments. You
can use the pre-trained version of the model and place it inside `experiments/`. To do so, download the experiment [from this link](https://www.dropbox.com/s/mv49j7c1hkqnncg/siamese_default.zip?dl=0) and execute:

```
mv siamese_default.zip experiments/
cd experiments/
unzip siamese_default.zip
```

This will create the folder `experiments/siamese_default`, where `siamese_default` is the name of the experiment that
will be asked during the execution of the cleaning script. You will also be asked about which checkpoint number you
want to use. We only provide the checkpoint of the last epoch, which corresponds to epoch `100`. You can create and
train your own experiments and models following the tutorials provided in
[Skeltorch documentation](https://docs.skeltorch.com/en/latest/).

Before cleaning the data, you will need to create a small test split. You will be asked to annotate a set of chunks,
deciding whether random chunks belong to the speaker or not. The number of samples to annotate will be asked during the
execution of the script. The more, the better.

With your annotated test split in hand, you will have to decide which threshold do you want to set to distinguish
between speakers. You can use the plots of the precision, recall and f-score to decide. As we are assigning the label
"positive" to samples which belong to different speakers, we want to minimize the number of false negatives in order to
minimize the number of chunks from other speakers. In terms of precision and recall, we should set a minimum value of
recall and maximize precision.

```
Usage:
    python -m speech_dataset_generator --dataset-name <experiment_name> clean_speakers

Example:
    $ python -m speech_dataset_generator --dataset-name donald_dataset clean_speakers
    $ How many test samples do you want to annotate (min. 10)? 10
    $ ==================================================
    $
    $ Speaker chunks in the list: 0/10
    $ Non-speaker chunks in the list: 0/10
    $ Chunk Text: Challenge will spur innovation and
    $ Is the chunk from your desired speaker? [y/n]: y
    $
    $ ==================================================
    $
    $ Speaker chunks in the list: 1/10
    $ Non-speaker chunks in the list: 0/10
    $ Chunk Text: County after County and state after
    $ Is the chunk from your desired speaker? [y/n]: n
    $
    $ [repeated until both lists are filled]
    $
    $ Which device do you want to use to run the model? [cpu/cuda] cpu
    $ What experiment do you want to use as Siamese network? (make sure it exists) siamese_default
    $ What checkpoint do you want to use? (make sure it exists) 100
    $
    $ [shows recall, precision and f-score plots using annotated data]
    $
    $ Which threshold do you want to set? 0.75
    $ How many comparisions per chunk do you want to perform? 16

```

## Contributing
You are invited to send your pull request if your contribution improves the current version of the project. Every
request will be carefully reviewed and considered.
