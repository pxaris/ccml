import os
import sys
import torch
import torch.nn as nn
import tqdm
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

# add project root to path
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(str(Path(current_dir)))

from config import MODELS_CONFIG, EVALUATIONS_DIR, DATA_DIR, MODELS_DIR
from training.datasets.utils import get_dataset_test_loader, split_spectrogram, get_dataset_train_val_loader, get_dataset_mean_std
from training.models.ast import ASTModel
from training.models.vgg_ish import ShortChunkCNN
from training.models.musicnn import Musicnn


def evaluate(config):
    result = [
        f'\nEvaluation of model "{config["model_name"]}" on "{config["dataset"]}" test set:']

    test_dataset, test_dataloader = get_dataset_test_loader(config)

    if 'normalize_input' in config and config['normalize_input']:
        _, train_loader, _ = get_dataset_train_val_loader(config)
        config['norm_mean'], config['norm_std'] = get_dataset_mean_std(
            train_loader)

    # load model
    saved_models_dir = os.path.join(MODELS_DIR, config['dataset'])
    model_path = os.path.join(saved_models_dir, f'{config["model_name"]}.pth')

    if 'musicnn' in config['model_name']:
        model = Musicnn(n_class=test_dataset.len_labels)
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(config['device'])))

    elif 'vgg_ish' in config['model_name']:
        model = ShortChunkCNN(n_class=test_dataset.len_labels)
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(config['device'])))

    elif 'ast' in config['model_name']:
        model = ASTModel(input_tdim=test_dataset.input_length,
                         label_dim=test_dataset.len_labels, model_size='base384')
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(config['device'])))
    else:
        raise NotImplementedError(
            'No model implementation found for the given config.')

    model = model.to(config['device'])
    model.eval()

    # get model prediction for each sample in the test set (store also its label)
    y = []
    y_ = []
    estimated = []
    sigmoid = torch.nn.Sigmoid()
    for single_sample_batch in tqdm.tqdm(test_dataloader):
        [mel_spectrogram], [label] = single_sample_batch
        splitted_spectrogram = split_spectrogram(
            mel_spectrogram, test_dataset.input_length)
        splits_scores = []
        for spectrogram in splitted_spectrogram:
            spectrogram = spectrogram[np.newaxis, :, :]

            if 'normalize_input' in config and config['normalize_input']:
                # normalize the input audio spectrogram so that the dataset mean
                # and standard deviation are 0 and 0.5 respectively (as in training)
                spectrogram = (
                    spectrogram - config['norm_mean']) / (config['norm_std'] * 2)

            out = model.forward(spectrogram.float().to(config['device']))
            splits_scores.append(out.detach().cpu().numpy())

        splits_scores = np.vstack(splits_scores)
        # average pooling on chunks scores
        tags_scores = np.mean(splits_scores, axis=0)

        if isinstance(config['loss_function'], nn.BCEWithLogitsLoss):
            # models with no sigmoid at their final layer when trained
            if isinstance(tags_scores, np.ndarray):
                tags_scores = torch.from_numpy(
                    tags_scores).float().to(config['device'])
            tags_scores = sigmoid(tags_scores)
            tags_scores = tags_scores.detach().cpu().numpy()

        prediction = np.around(tags_scores)

        y_.append(prediction)
        y.append(label.detach().numpy())
        estimated.append(tags_scores)

    # roc_auc_score, pr_auc_score
    result += [f'ROC-AUC score: {roc_auc_score(y, estimated)}']
    result += [f'PR-AUC score: {average_precision_score(y, estimated)}\n']

    # detailed report
    if hasattr(test_dataset, 'label_transformer'):
        target_names = [
            str(label) for label in test_dataset.label_transformer.classes_.tolist()]
        result += [classification_report(y, y_, target_names=target_names)]
    else:
        result += [classification_report(y, y_)]

    return result


def run_evaluation(config):
    print('Evaluation started...')
    evaluation_result = []
    evaluation_result += evaluate(config)

    output_dir = os.path.join(EVALUATIONS_DIR, args.dataset)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_filename = f'{args.model_name}.txt'
    output_file = os.path.join(output_dir, output_filename)

    with open(output_file, 'w', encoding='utf-8') as f:
        for value in evaluation_result:
            f.write(str(value) + '\n')
    print(f'\tDone. Values were written to file {output_file}')


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
    run_evaluation(config)
