import os
import torch.nn as nn

# dir paths used in several places
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
EVALUATIONS_DIR = os.path.join(ROOT_DIR, 'evaluation')

# audio and mel-spectrograms attributes
SPECTROGRAMS_ATTRIBUTES = {
    'audio_sr': 16000,
    'n_fft': 512,
    'hop_length': 256,
    'f_min': 0.0,
    'f_max': 8000.0,
    'n_mels': 128,
}

# total duration of some datasets was decreased
# by setting max audio duration in seconds
MAX_AUDIO_DURATION = {
    'magnatagatune': None,
    'fma': None,
    'lyra': None,
    'makam': 150,
    'hindustani': 780,
    'carnatic': 330,
}

MODELS_CONFIG = {
    'magnatagatune': {
        'musicnn': {
            'input_length_in_secs': 3,
            'LR': 1e-4,
            'epochs': 50,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'vgg_ish': {
            'input_length_in_secs': 3.69,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'ast': {
            'input_length_in_secs': 8,
            'LR': 1e-5,
            'epochs': 50,
            'batch_size': 12,
            'early_stopping_patience': None,
            'normalize_input': True,
            'loss_function': nn.BCEWithLogitsLoss(),
            'optimizer_type': 'adam_lr_scheduler_warmup',
        },
    },
    'fma': {
        'musicnn': {
            'input_length_in_secs': 3,
            'top_N_tags': 20,
            'LR': 1e-4,
            'epochs': 50,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'vgg_ish': {
            'input_length_in_secs': 3.69,
            'top_N_tags': 20,
            'LR': 1e-4,
            'epochs': 100,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'ast': {
            'input_length_in_secs': 8,
            'top_N_tags': 20,
            'LR': 1e-5,
            'epochs': 50,
            'batch_size': 12,
            'early_stopping_patience': None,
            'normalize_input': True,
            'loss_function': nn.BCEWithLogitsLoss(),
            'optimizer_type': 'adam_lr_scheduler_warmup',
        },
    },
    'lyra': {
        'musicnn': {
            'input_length_in_secs': 3,
            'top_N_tags': 30,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'validation_size': 0.1,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'vgg_ish': {
            'input_length_in_secs': 3.69,
            'top_N_tags': 30,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'validation_size': 0.1,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'ast': {
            'input_length_in_secs': 8,
            'top_N_tags': 30,
            'LR': 1e-5,
            'epochs': 50,
            'batch_size': 12,
            'validation_size': 0.1,
            'early_stopping_patience': None,
            'normalize_input': True,
            'loss_function': nn.BCEWithLogitsLoss(),
            'optimizer_type': 'adam_lr_scheduler_warmup',
        },
    },
    'makam': {
        'musicnn': {
            'input_length_in_secs': 3,
            'top_N_tags': 30,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'vgg_ish': {
            'input_length_in_secs': 3.69,
            'top_N_tags': 30,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'ast': {
            'input_length_in_secs': 8,
            'top_N_tags': 30,
            'LR': 1e-5,
            'epochs': 70,
            'batch_size': 12,
            'early_stopping_patience': None,
            'normalize_input': True,
            'loss_function': nn.BCEWithLogitsLoss(),
            'optimizer_type': 'adam_lr_scheduler_warmup',
        },
    },
    'hindustani': {
        'musicnn': {
            'input_length_in_secs': 3,
            'top_N_tags': 20,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'vgg_ish': {
            'input_length_in_secs': 3.69,
            'top_N_tags': 20,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'ast': {
            'input_length_in_secs': 8,
            'top_N_tags': 20,
            'LR': 1e-5,
            'epochs': 50,
            'batch_size': 12,
            'early_stopping_patience': None,
            'normalize_input': True,
            'loss_function': nn.BCEWithLogitsLoss(),
            'optimizer_type': 'adam_lr_scheduler_warmup',
        },
    },
    'carnatic': {
        'musicnn': {
            'input_length_in_secs': 3,
            'top_N_tags': 20,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'vgg_ish': {
            'input_length_in_secs': 3.69,
            'top_N_tags': 20,
            'LR': 1e-4,
            'epochs': 200,
            'batch_size': 16,
            'early_stopping_patience': None,
            'loss_function': nn.BCELoss(),
            'optimizer_type': 'scheduler',
        },
        'ast': {
            'input_length_in_secs': 8,
            'top_N_tags': 20,
            'LR': 1e-5,
            'epochs': 70,
            'batch_size': 12,
            'early_stopping_patience': None,
            'normalize_input': True,
            'loss_function': nn.BCEWithLogitsLoss(),
            'optimizer_type': 'adam_lr_scheduler_warmup',
        },
    }
}

# add Transfer Learning configs to models config dict
datasets = MODELS_CONFIG.keys()
models = ['vgg_ish', 'musicnn', 'ast']

for target in datasets:
    for model in models:
        for source in datasets:
            if source == target:
                continue
            for freeze in [True, False]:
                # freeze all network except final layer if True
                model_name = f'{model}_from_{source}_f' if freeze else f'{model}_from_{source}'
                # load target dataset respective model config
                MODELS_CONFIG[target][model_name] = MODELS_CONFIG[target][model].copy()
                # add transfer learning configuration
                MODELS_CONFIG[target][model_name]['transfer'] = {
                    'source_model': os.path.join(MODELS_DIR, source, f'{model}.pth'),
                    'freeze': freeze,
                    'final_layer': 'mlp_head.1' if model == 'ast' else 'dense2',
                }
