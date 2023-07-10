import torch
import numpy as np
from training.datasets.lyra import LyraDataset, torch_train_val_split
from training.datasets.magnatagatune import Magnatagatune
from training.datasets.ccml_dataset import CCMLDataset
from torch.utils.data import DataLoader


def get_dataset_train_val_loader(config):

    if config['dataset'] == 'lyra':
        train_dataset = LyraDataset(
            config['data_dir'],
            top_N=config['top_N_tags'],
            input_length_in_secs=config['input_length_in_secs'],
            train=True,
        )
        train_loader, val_loader = torch_train_val_split(
            train_dataset, config['batch_size'], config['validation_size'])

    elif config['dataset'] == 'magnatagatune':
        train_dataset = Magnatagatune(
            config['data_dir'],
            input_length_in_secs=config['input_length_in_secs'],
            keyword='train',
        )
        train_loader = get_data_loader(train_dataset, config['batch_size'])

        val_dataset = Magnatagatune(
            config['data_dir'],
            input_length_in_secs=config['input_length_in_secs'],
            keyword='valid',
        )
        val_loader = get_data_loader(val_dataset, config['batch_size'])
    
    elif config['dataset'] in ['fma', 'makam', 'carnatic', 'hindustani']:
        train_dataset = CCMLDataset(
            config['data_dir'],
            top_N=config['top_N_tags'],
            input_length_in_secs=config['input_length_in_secs'],
            keyword='train',
        )
        train_loader = get_data_loader(train_dataset, config['batch_size'])

        val_dataset = CCMLDataset(
            config['data_dir'],
            top_N=config['top_N_tags'],
            input_length_in_secs=config['input_length_in_secs'],
            keyword='valid',
        )
        val_loader = get_data_loader(val_dataset, config['batch_size'])

    else:
        raise NotImplementedError(
            f'No implementation found for dataset {config["dataset"]}')

    return train_dataset, train_loader, val_loader


def get_dataset_test_loader(config):

    if config['dataset'] == 'lyra':
        test_dataset = LyraDataset(
            config['data_dir'],
            top_N=config['top_N_tags'],
            input_length_in_secs=config['input_length_in_secs'],
            train=False,
        )

    elif config['dataset'] == 'magnatagatune':
        test_dataset = Magnatagatune(
            config['data_dir'],
            input_length_in_secs=config['input_length_in_secs'],
            keyword='test',
        )
    
    elif config['dataset'] in ['fma', 'makam', 'carnatic', 'hindustani']:
        test_dataset = CCMLDataset(
            config['data_dir'],
            top_N=config['top_N_tags'],
            input_length_in_secs=config['input_length_in_secs'],
            keyword='test',
        )

    else:
        raise NotImplementedError(
            f'No implementation found for dataset {config["dataset"]}')

    test_loader = get_data_loader(test_dataset, 1, shuffle=False)
    return test_dataset, test_loader


def get_data_loader(dataset, batch_size, shuffle=True):
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader


def split_spectrogram(spectrogram, split_length, keep_residual=False):
    spectr_length = spectrogram.shape[0]
    num_spectrs, residual_length = int(
        spectr_length/split_length), spectr_length % split_length

    if num_spectrs == 0:
        return None

    specgram_to_split = spectrogram[:-
                                    residual_length] if residual_length else spectrogram
    splitted_spectrogram = np.split(specgram_to_split, num_spectrs)
    if keep_residual and residual_length:
        splitted_spectrogram += [spectrogram[-residual_length:]]

    return splitted_spectrogram


def get_dataset_mean_std(dataloader):
    running_sum, running_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        running_sum += torch.mean(data)
        running_squared_sum += torch.mean(data**2)
        num_batches += 1

    mean = running_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (running_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
