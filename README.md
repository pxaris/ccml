# Cross-cultural Music Transfer learning

PyTorch implementation of cross-cultural music transfer learning using auto-tagging models.

## Reference

[**From West to East: Who can understand the music of the others better?**](https://arxiv.org/abs/2307.09795), ISMIR 2023.  
- Charilaos Papaioannou, Emmanouil Benetos, and Alexandros Potamianos

## Datasets

- [**MagnaTagATune**](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
  - audios: they can be downloaded from the dataset webpage.
  - metadata: the (binary) labels for the top-50 tags are stored in `data/magnatagatune/split/binary.npy` and they can be also found at [minzwon/sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models/tree/master/split/mtat) repository.
- [**FMA**](https://github.com/mdeff/fma)
  - audios: they can be downloaded from the dataset repository for the FMA-medium dataset that is used in this study.
  - metadata: the top-20 hierarchically related genre labels (available in dataset repository) were utilized to create the `data/fma/split/metadata.npy` file which contains a dictionary where keys are the track ids and values are the repective labels. 
- [**Lyra**](https://github.com/pxaris/lyra-dataset)
  - audios: they are not publicly available but the mel-spectrograms can be downloaded from the dataset repository.
  - metadata: they have been copied to the directory `data/lyra/split/` from the dataset repository.
- [**Turkish-makam**](https://dunya.compmusic.upf.edu/makam/), [**Hindustani**](https://dunya.compmusic.upf.edu/hindustani/), [**Carnatic**](https://dunya.compmusic.upf.edu/carnatic/)
  - These datasets are part of the [CompMusic Corpora](https://compmusic.upf.edu/corpora). One should create an account to [Dunya](https://dunya.compmusic.upf.edu/) and request access to the audio files. 
  - The [pycompmusic](https://github.com/MTG/pycompmusic) has to be installed and the user `token` must be used at the helper scripts under `preprocessing/dunya/` directory. In order to fetch both data and metadata, set the `dataset` option properly (one of `'makam', 'hindustani', 'carnatic'`) and execute the scripts such as:
```bash
python preprocessing/dunya/get_audios.py --dataset 'makam'
python preprocessing/dunya/get_metadata.py --dataset 'carnatic'
```

The metadata are supposed to be under `data/{dataset}/split/` directory at the files `binary.npy` for MagnaTagATune, `training.tsv` and `test.tsv` for Lyra and `metadata.npy` for the rest of the datasets.

Regarding **splits**, the analogies 0.7/0.1/0.2 for training, validation and test sets are used across all datasets. Specifically: 
- MagnaTagATune: the publicly available split is used. It can be found in at several repositories, e.g. [here](https://github.com/minzwon/sota-music-tagging-models/tree/master/split/mtat).
- Lyra: The [publicly available split](https://github.com/pxaris/lyra-dataset/tree/main/data/split) for training and test sets is used. The validation set is randomly splitted from the training set during training.
- For the rest of the datasets - FMA-medium, Turkish-makam, Hindustani and Carnatic - random split was applied and the result was stored in the files `train.npy`, `valid.npy` and `test.npy` under each `data/{dataset}/split` directory, for reproducibility. Each one of those files contains a **list of ids** that is used, in turn, by the respective dataloader. 

## Requirements

* Python 3.8 or later
* Create virtual environment and install requirements
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Preprocessing

The audio (mp3) files of each dataset are expected to be found under `audios/` directory at a specific path (let's say `{data_dir}`) such as `/__path_to__/data/{dataset}/`. 

In order to create the `mel-spectrograms` for all datasets except Lyra (for which they can be readily downloaded), the following command can be used by setting properly the `dataset` (use one of `'magnatagatune', 'fma', 'makam', 'hindustani', 'carnatic'`) and `data_dir` (full path to the dataset directory) options, such as:

```bash
python preprocessing/create_mel_spectrograms.py --dataset 'magnatagatune' --data_dir '/__path_to__/magnatagatune'
```

The mel-spectrograms will be generated under the respective `{data_dir}/mel-spectrograms/` directory and they will follow the `{id}.npy` naming convention.

## Training

The available deep learning models are [**VGG-ish**](https://arxiv.org/abs/2006.00751), [**Musicnn**](https://arxiv.org/abs/1711.02520) and [**Audio Spectrogram Transformer (AST)**](https://arxiv.org/abs/2104.01778). 

Specify:
- `{dataset}` as one of `'magnatagatune', 'fma', 'makam', 'lyra', 'hindustani', 'carnatic'`
- `{data_dir}` where the `mel-spectrograms` and `split` dirs are expected to be found
- `{model_name}` which will load the respective configuration from `MODELS_CONFIG` at `config.py`
  - use one of `'vgg_ish', 'musicnn', 'ast'` for models that no transfer learning is taking place
- `{device}` to be used (one of `'cpu', 'cuda:0', 'cuda:1'` etc.)

example:
```bash
python train.py --dataset 'magnatagatune' --data_dir '/__path_to__/magnatagatune' --model_name 'ast' --device 'cuda:0'
```

### Transfer Learning

Once training of a model has been completed on a dataset, it can be used to apply transfer learning to another by using the following conventions for the `{model_name}` option:
- `{model}_from_{dataset}` when fine-tuning on the whole network will be applied
- `{model}_from_{dataset}_f` when fine-tuning only of the final layer will take place

Assuming that a `vgg_ish` model is trained on `fma` and one wishes to transfer it to `hindustani` and fine-tune the whole network, the command will be:
```bash
python train.py --dataset 'hindustani' --data_dir '/__path_to__/hindustani' --model_name 'vgg_ish_from_fma' --device 'cuda:0'
```

In order to transfer a `musicnn` model from `'lyra'` to `'makam'` dataset and fine-tune only the final layer, one would run: 
```bash
python train.py --dataset 'makam' --data_dir '/__path_to__/makam' --model_name 'musicnn_from_lyra_f' --device 'cuda:0'
```

## Evaluation

For evaluating a model, the same options need to be specified, i.e. `{dataset}`, `{data_dir}`, `{model_name}` and `{device}`.

example for a single-domain model:
```bash
python evaluate.py --dataset 'fma' --data_dir '/__path_to__/fma' --model_name 'ast' --device 'cuda:0'
```

or for a transfer learning model:
```bash
python evaluate.py --dataset 'lyra' --data_dir '/__path_to__/lyra' --model_name 'musicnn_from_magnatagatune' --device 'cuda:0'
```

The result will be stored under `evaluation/{dataset}/` directory to a `{model_name}.txt` file. The evaluation results of the single-domain and the best transfer learning models are being provided for reference purposes.   
