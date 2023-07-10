import os
import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from config import SPECTROGRAMS_ATTRIBUTES


class LyraDataset(Dataset):

    def __init__(self,
                 data_path,
                 top_N=30,
                 mel_specs_config=SPECTROGRAMS_ATTRIBUTES,
                 input_length_in_secs=3,
                 train=True,
                 ):

        self.mel_specs_config = mel_specs_config
        self.top_N = top_N

        self.input_length_in_secs = input_length_in_secs
        self.input_length = int(self.input_length_in_secs *
                                self.mel_specs_config['audio_sr']/self.mel_specs_config['hop_length'])

        self.data_path = data_path
        keyword = "training" if train else "test"
        annotations_file = os.path.join(
            data_path, 'split', keyword + '.tsv')
        self.specs_dir = os.path.join(data_path, 'mel-spectrograms')

        self.ids, self._labels = self.get_ids_labels(annotations_file)
        if self.top_N:
            if train:
                labels_merged = sum(self._labels, [])
            else:
                # use top_N tags of the training set while on test mode as well
                training_annotations_file = os.path.join(
                    data_path, 'split', 'training.tsv')
                _, train_labels = self.get_ids_labels(
                    training_annotations_file)
                labels_merged = sum(train_labels, [])

            tags_frequency = {key: len(list(group))
                              for key, group in groupby(np.sort(labels_merged))}
            tags_frequency_sorted = {k: v for k, v in sorted(
                tags_frequency.items(), key=lambda item: item[1], reverse=True)}
            self._tags = list(tags_frequency_sorted.keys())[:self.top_N]

        else:
            labels_merged = sum(self._labels, [])
            self._tags = list(set(labels_merged))

        self.mel_specs = []
        self.labels = []
        for piece_id, label_list in zip(self.ids, self._labels):
            spectrogram_file = os.path.join(self.specs_dir, piece_id + '.npy')
            if train:
                piece_mel = self.get_random_spec_segment_from_file(
                        spectrogram_file)
            else:
                piece_mel = self.get_spectrogram_from_file(spectrogram_file)
            
            self.mel_specs += piece_mel
            target_label_list = list(set(label_list) & set(self._tags))
            self.labels += [target_label_list]

        self.len_labels = len(self._tags)

        self.label_transformer = MultiLabelBinarizer(classes=self._tags)
        self.labels = np.array(
            self.label_transformer.fit_transform(self.labels)
        ).astype("int64")

    def get_random_spec_segment_from_file(self, spectrogram_file):
        spectrogram = np.load(spectrogram_file).T
        random_idx = int(np.floor(np.random.random(
            1) * (len(spectrogram)-self.input_length)))
        return [np.array(spectrogram[random_idx:random_idx+self.input_length])]

    @staticmethod
    def get_spectrogram_from_file(spectrogram_file):
        return [np.load(spectrogram_file).T]

    def get_ids_labels(self, annotations_file):
        data_df = pd.read_csv(annotations_file, sep='\t')
        ids, labels = [], []
        for _, row in data_df.iterrows():
            ids.append(row['id'])
            label_list = []
            for col_name in ['instruments', 'genres', 'place']:
                label_list += [
                    f'{col_name}--{label}' for label in row[col_name].split('|')]
                        
            labels.append(label_list)
        
        return ids, labels

    def __getitem__(self, item):
        return self.mel_specs[item], self.labels[item]

    def __len__(self):
        return len(self.labels)


def torch_train_val_split(dataset, batch_size, val_size=0.1, shuffle=True, seed=10):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PyTorch data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader


def get_test_loader(test_dataset):
    return DataLoader(test_dataset)
