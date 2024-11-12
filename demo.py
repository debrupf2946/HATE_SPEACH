from hate_speech.logger import logging
from hate_speech.configuration.gcloud_syncer import GCloudSync

obj=GCloudSync()

obj.sync_folder_from_gcloud('hate_speech_2026', 'dataset.zip', 'download/')