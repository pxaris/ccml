import os
import sys
import argparse
import os
import sys
from pathlib import Path

# add project root to path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(current_dir).parent))

from config import DATA_DIR, SPECTROGRAMS_ATTRIBUTES, MAX_AUDIO_DURATION
from preprocessing.utils import create_mel_spectrograms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='magnatagatune', choices=[
                        'magnatagatune', 'fma', 'makam', 'hindustani', 'carnatic'])
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(DATA_DIR, 'magnatagatune'))
    args = parser.parse_args()

    config = SPECTROGRAMS_ATTRIBUTES
    config['data_dir'] = args.data_dir
    config['max_audio_duration'] = MAX_AUDIO_DURATION[args.dataset]

    mel_specs_dir = os.path.join(config['data_dir'], 'mel-spectrograms')
    Path(mel_specs_dir).mkdir(parents=True, exist_ok=True)

    create_mel_spectrograms(config, mel_specs_dir)
