import os
import progressbar
import requests
import time


class CreateList:
    api_endpoint = 'https://www.googleapis.com/youtube/v3/search'
    url_template = 'https://www.youtube.com/watch?v={}'
    dataset_path = None
    dataset_list_path = None
    videos_list = None

    def __init__(self, dataset_name):
        root_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data'
        )
        self.dataset_path = os.path.join(root_path, dataset_name)
        self.dataset_list_path = os.path.join(self.dataset_path, 'list.txt')

    def run(self):
        # Request user input: search query
        search_query = None
        while search_query is None or search_query == '':
            search_query = input(
                'What keyword do you want to use to find related videos? '
            )

        # Request user input: maximum videos to list
        max_videos = None
        while max_videos is None or not max_videos.isdigit() or \
                int(max_videos) < 1:
            max_videos = input('How many videos do you want to list? ')
        max_videos = int(max_videos)

        # Request user input: YouTube Data API key
        api_key = None
        while api_key is None or api_key == '':
            api_key = input('Introduce your YouTube Data API Key? ')

        # Initialize pipeline
        self._initialize()

        # Call the API until the list of videos is full
        self.videos_list = []
        page_token = None
        bar = progressbar.ProgressBar(max_value=max_videos)
        while len(self.videos_list) < max_videos:
            videos_ids, page_token = self._call_api(
                search_query, api_key, page_token
            )
            self.videos_list += [
                self.url_template.format(video_id) for video_id in videos_ids
            ]
            bar.update(min(len(self.videos_list), max_videos))
            time.sleep(1)
        self.videos_list = self.videos_list[:max_videos]

        # Save result as text file
        with open(self.dataset_list_path, 'w') as f:
            for item in self.videos_list:
                f.write('{}\n'.format(item))

    def _initialize(self):
        if os.path.exists(self.dataset_path):
            raise NameError('A data set with that name already exists')
        else:
            os.makedirs(self.dataset_path)

    def _call_api(self, search_query, api_key, page_token=None):
        response_raw = requests.get(
            url=self.api_endpoint,
            params={'key': api_key, 'part': 'id', 'q': search_query,
                    'type': 'video', 'eventType': 'completed',
                    'maxResults': 50, 'pageToken': page_token}
        )
        if not response_raw.ok:
            raise ConnectionError('Not possible to retrieve YouTube results.')
        response_json = response_raw.json()
        videos_ids = [
            item['id']['videoId']
            for item in response_json['items']
            if item['id']['kind'] == 'youtube#video'
        ]
        return videos_ids, response_json['nextPageToken']
