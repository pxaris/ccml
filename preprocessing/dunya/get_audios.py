import os
import sys
import argparse
from tqdm import tqdm
from pathlib import Path
from compmusic import dunya

# add project root to path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(current_dir).parent.parent))

from config import DATA_DIR


MY_DUNYA_TOKEN = '___MY_TOKEN__'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='makam',
                        choices=['makam', 'hindustani', 'carnatic'])
    args = parser.parse_args()

    print(f'Started audios downloading for dataset {args.dataset}...')

    dunya.set_token(MY_DUNYA_TOKEN)

    if args.dataset == 'makam':
        recordings = dunya.makam.get_recordings(recording_detail=True)
    elif args.dataset == 'hindustani':
        recordings = dunya.hindustani.get_recordings(recording_detail=True)
    elif args.dataset == 'carnatic':
        recordings = dunya.carnatic.get_recordings(recording_detail=True)
    else:
        raise NotImplementedError(f'Unknown dataset: {args.dataset}')

    audios_dir = os.path.join(DATA_DIR, args.dataset, 'audios')
    Path(audios_dir).mkdir(parents=True, exist_ok=True)

    for recording in tqdm(recordings):
        file_path = os.path.join(audios_dir, f'{recording["mbid"]}.mp3')
        if os.path.isfile(file_path):
            print('File exists, skipping...')
            continue
        try:
            mp3_content = dunya.docserver.get_mp3(recording['mbid'])
            with open(file_path, "wb") as f:
                f.write(mp3_content)
        except:
            print(
                f'An error occurred when downloading the recording {recording["mbid"]}')
