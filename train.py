import os
import sys
import torch
import argparse
import torch.nn as nn
from pathlib import Path

# add project root to path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(current_dir)))

from config import MODELS_CONFIG, DATA_DIR, MODELS_DIR
from training.datasets.utils import get_dataset_train_val_loader, get_dataset_mean_std
from training.models.ast import ASTModel
from training.models.vgg_ish import ShortChunkCNN
from training.models.musicnn import Musicnn
from training.utils import train_model


def run_training(config):

    train_dataset, train_loader, val_loader = get_dataset_train_val_loader(
        config)
    print(
        f'training set: {len(train_loader)} batches\nvalidation set: {len(val_loader)} batches')

    if 'normalize_input' in config and config['normalize_input']:
        config['norm_mean'], config['norm_std'] = get_dataset_mean_std(
            train_loader)

    # get the 1st batch values of the data loader
    x_b1, y_b1 = next(iter(train_loader))
    n_labels = y_b1[0].shape[0]

    # model
    if 'musicnn' in config['model_name']:
        model = Musicnn(n_class=n_labels)
    elif 'vgg_ish' in config['model_name']:
        model = ShortChunkCNN(n_class=n_labels)
    elif 'ast' in config['model_name']:
        model = ASTModel(input_tdim=train_dataset.input_length,
                         label_dim=n_labels, model_size='base384')
    else:
        raise NotImplementedError(
            f'No model specified for name: {config["model_name"]}')

    model.to(config['device'])

    # transfer learning
    if 'transfer' in config:
        source_model = torch.load(config['transfer']['source_model'], map_location=torch.device(config['device']))
        # remove weights of output layer
        del source_model[f'{config["transfer"]["final_layer"]}.weight']
        del source_model[f'{config["transfer"]["final_layer"]}.bias']
        # initialize target model with the state of source model
        model.load_state_dict(source_model, strict=False)
        # freeze all the network except the final layer
        if config['transfer']['freeze']:
            final_layer_params = [f'{config["transfer"]["final_layer"]}.weight', f'{config["transfer"]["final_layer"]}.bias']
            for name, param in model.named_parameters():
                if name not in final_layer_params:
                    param.requires_grad = False

    # training setup
    if not config['loss_function']:
        config['loss_function'] = nn.BCEWithLogitsLoss()

    if config['optimizer_type'] == 'scheduler':
        config['optimizer'] = torch.optim.Adam(
            model.parameters(), config['LR'], weight_decay=1e-4)
    elif config['optimizer_type'] == 'adam_lr_scheduler_warmup':
        config['optimizer'] = torch.optim.Adam(
            model.parameters(), config['LR'], weight_decay=5e-7, betas=(0.95, 0.999))
        config['lr_scheduler'] = torch.optim.lr_scheduler.MultiStepLR(
            config['optimizer'], list(range(5, 1000, 1)), gamma=0.85)
        config['global_step'] = 0
    else:
        raise NotImplementedError(
            f'No optimizer specified for type: {config["optimizer_type"]}')

    model_filename = f'{config["model_name"]}.pth'
    saved_models_dir = os.path.join(MODELS_DIR, config['dataset'])
    Path(saved_models_dir).mkdir(parents=True, exist_ok=True)

    config['save_path'] = os.path.join(saved_models_dir, model_filename)

    train_model(model, train_loader, val_loader, config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=str, default='magnatagatune', choices=[
                        'magnatagatune', 'fma', 'lyra', 'makam', 'hindustani', 'carnatic'])
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(DATA_DIR, 'magnatagatune'))
    parser.add_argument('--model_name', type=str, default='musicnn')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'])
    args = parser.parse_args()

    config = MODELS_CONFIG[args.dataset][args.model_name].copy()
    config['dataset'] = args.dataset
    config['data_dir'] = args.data_dir
    config['model_name'] = args.model_name
    config['device'] = torch.device(args.device)

    print(config)
    run_training(config)
