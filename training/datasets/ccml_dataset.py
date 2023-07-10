import os
import numpy as np
from itertools import groupby
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from config import SPECTROGRAMS_ATTRIBUTES


class CCMLDataset(Dataset):
    def __init__(self,
                 data_path,
                 top_N=None,
                 mel_specs_config=SPECTROGRAMS_ATTRIBUTES,
                 input_length_in_secs=3,
                 keyword='train',
                 ):
        self.data_path = data_path
        self.top_N = top_N
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

        self.ids = np.load(os.path.join(
            self.split_path, f'{keyword}.npy'), allow_pickle=True).tolist()
        self.metadata = np.load(os.path.join(
            self.split_path, 'metadata.npy'), allow_pickle=True).item()
        self._labels = [self.metadata[id] for id in self.ids]

        if self.top_N:
            if keyword == 'train':
                labels_merged = sum(self._labels, [])
            else:
                # use top_N tags of the training set while on test/valid mode as well
                train_ids = np.load(os.path.join(
                    self.split_path, 'train.npy'), allow_pickle=True).tolist()
                train_labels = [self.metadata[id] for id in train_ids]
                labels_merged = sum(train_labels, [])

            tags_frequency = {key: len(list(group))
                              for key, group in groupby(np.sort(labels_merged))}
            tags_frequency_sorted = {k: v for k, v in sorted(
                tags_frequency.items(), key=lambda item: item[1], reverse=True)}
            self._tags = list(tags_frequency_sorted.keys())[:self.top_N]

        else:
            labels_merged = sum(self._labels, [])
            self._tags = list(set(labels_merged))

        self.len_labels = len(self._tags)
        self.labels = [list(set(label_list) & set(self._tags)) for label_list in self._labels]

        self.label_transformer = MultiLabelBinarizer(classes=self._tags)
        self.labels = np.array(
            self.label_transformer.fit_transform(self.labels)
        ).astype("int64")

    def __getitem__(self, item):
        spectrogram, tags_binary = self.get_spectrogram_and_tags(item)
        return spectrogram.astype('float32'), tags_binary.astype('float32')

    def get_spectrogram_and_tags(self, item):
        spec_path = os.path.join(self.specs_dir, f'{self.ids[item]}.npy')
        spectrogram = np.load(spec_path, mmap_mode='r').T

        if self.keyword in ['train', 'valid']:
            # get a single chunk randomly
            random_idx = int(np.floor(np.random.random(1) *
                                      (len(spectrogram)-self.input_length)))
            spectrogram = np.array(
                spectrogram[random_idx:random_idx+self.input_length])

        tags_binary = self.labels[item]

        return spectrogram, tags_binary

    def __len__(self):
        return len(self.ids)
