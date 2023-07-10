import os
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from config import SPECTROGRAMS_ATTRIBUTES


class Magnatagatune(Dataset):
    def __init__(self,
                 data_path,
                 mel_specs_config=SPECTROGRAMS_ATTRIBUTES,
                 input_length_in_secs=3,
                 keyword='train',
                 ):
        self.data_path = data_path
        self.specs_dir = os.path.join(data_path, 'mel-spectrograms')
        self.mel_specs_config = mel_specs_config

        self.input_length_in_secs = input_length_in_secs
        self.input_length = int(self.input_length_in_secs *
                                self.mel_specs_config['audio_sr']/self.mel_specs_config['hop_length'])

        self.split_path = os.path.join(data_path, 'split')
        self.keyword = keyword

        if keyword not in ['train', 'test', 'valid']:
            raise ValueError(
                f'keyword must take one of the values: "train", "test", "valid"')

        self.files_list = np.load(os.path.join(
            self.split_path, f'{keyword}.npy'))
        self.tags_binary = np.load(os.path.join(self.split_path, 'binary.npy'))
        self.len_labels = len(self.tags_binary[0])

    def __getitem__(self, item):
        spectrogram, tags_binary = self.get_spectrogram_and_tags(item)
        return spectrogram.astype('float32'), tags_binary.astype('float32')

    def get_spectrogram_and_tags(self, item):
        index, file_path = self.files_list[item].split('\t')

        spec_path = os.path.join(self.specs_dir, f'{Path(file_path).stem}.npy')
        spectrogram = np.load(spec_path, mmap_mode='r').T

        if self.keyword in ['train', 'valid']:
            # get a single chunk randomly
            random_idx = int(np.floor(np.random.random(1) *
                                    (len(spectrogram)-self.input_length)))
            spectrogram = np.array(
                spectrogram[random_idx:random_idx+self.input_length])
        
        tags_binary = self.tags_binary[int(index)]

        return spectrogram, tags_binary

    def __len__(self):
        return len(self.files_list)
