import os
import sys
import argparse
import numpy as np
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

    parser.add_argument('--dataset', type=str, default='makam', choices=['makam', 'hindustani', 'carnatic'])
    args = parser.parse_args()

    print(f'Started metadata file creation for dataset {args.dataset}...')
    
    dunya.set_token(MY_DUNYA_TOKEN)
    
    # load all ids
    split_dir = os.path.join(DATA_DIR, args.dataset, 'split')
    train_ids = np.load(os.path.join(split_dir, 'train.npy'), allow_pickle=True).tolist()
    valid_ids = np.load(os.path.join(split_dir, 'valid.npy'), allow_pickle=True).tolist()
    test_ids = np.load(os.path.join(split_dir, 'test.npy'), allow_pickle=True).tolist()
    ids = train_ids + valid_ids + test_ids
    
    metadata = {}
    for id in tqdm(ids):
        
        if args.dataset == 'makam':
            try:
                recording = dunya.makam.get_recording(id)
                makam_list = [f'makam--{item["name"]}' for item in recording['makamlist']]
                usul_list = [f'usul--{item["name"]}' for item in recording['usullist']]
                instruments_list = [f"instrument--{item['instrument']['name']}" for item in recording['performers']]
                tags = list(set(makam_list + usul_list + instruments_list))
            except:
                tags = None
        
        elif args.dataset == 'carnatic':
            try:
                recording = dunya.carnatic.get_recording(id)
                raga_list = [f'raga--{item["common_name"]}' for item in recording['raaga']]
                tala_list = [f'tala--{item["common_name"]}' for item in recording['taala']]
                instruments_list = [f"instrument--{item['instrument']['name']}" for item in recording['artists']]
                form_list = [f"form--{item['name']}" for item in recording['form']]
                tags = list(set(raga_list + tala_list + instruments_list + form_list))
            except:
                tags = None
        
        elif args.dataset == 'hindustani':
            try:
                recording = dunya.hindustani.get_recording(id)
                raga_list = [f'raga--{item["common_name"]}' for item in recording['raags']]
                tala_list = [f'tala--{item["common_name"]}' for item in recording['taals']]
                instruments_list = [f"instrument--{item['instrument']['name']}" for item in recording['artists']]
                form_list = [f"form--{item['common_name']}" for item in recording['forms']]
                tags = list(set(raga_list + tala_list + instruments_list + form_list))
            except:
                tags = None
        
        metadata[id] = tags

    metadata_file = os.path.join(split_dir, 'metadata.npy')
    np.save(metadata_file, metadata)
    print(f'\tDone. Metadata file created at: {metadata_file}')
