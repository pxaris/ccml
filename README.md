# Cross-cultural Music Transfer learning

PyTorch implementation of cross-cultural music transfer learning using auto-tagging models.

## Reference

**From West to East: Who can understand the music of the others better?**, ISMIR 2023.  
- Charilaos Papaioannou, Emmanouil Benetos, and Alexandros Potamianos

## Datasets
- [**MagnaTagATune**](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
- [**FMA**](https://github.com/mdeff/fma)
- [**Lyra**](https://github.com/pxaris/lyra-dataset)
- [**Turkish-makam**](https://dunya.compmusic.upf.edu/makam/)
- [**Hindustani**](https://dunya.compmusic.upf.edu/hindustani/)
- [**Carnatic**](https://dunya.compmusic.upf.edu/carnatic/)

## Models
- **VGG-ish** : CNN architecture with multiple layers based on the VGG network, as implemented by [Won et al.](https://arxiv.org/abs/2006.00751)
- **Musicnn** : Music inspired model that uses convolutional layers at its core, [Pons et al.](https://arxiv.org/abs/1711.02520)
- **Audio Spectrogram Transformer** : Purely attention-based model for audio classification, [Gong et al.](https://arxiv.org/abs/2104.01778)

## Data preparation

All models use mel-spectrograms for their input. For each dataset the files with the naming convention `{id}.npy` are expected to be found under `data/{dataset}/mel-spectrograms/` directory. As for the tags, they are supposed to be in the `data/{dataset}/split/metadata.npy` file, except for Lyra where the `{training|test}.tsv` files are used.

### Sources

- MagnaTagATune
  - audios: they can be downloaded from the [dataset webpage](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset).
  - metadata: the (binary) labels for the top-50 tags are stored in `data/magnatagatune/split/binary.npy` and they can be found also at [minzwon/sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models/tree/master/split/mtat) repository.
- FMA
  - audios: they can be downloaded from the [dataset repository](https://github.com/mdeff/fma) for the FMA-medium dataset that is used in this study.
  - metadata: the top-20 hierarchically related genres annotation (available in dataset repository) was utilized to create the `data/fma/split/metadata.npy` file which contains a dictionary where keys are the track ids and values are the repective labels. 
- Lyra
  - audios: they are not publicly available but the mel-spectrograms can be downloaded from the [dataset repository](https://github.com/pxaris/lyra-dataset)
  - metadata: they have been copied to the directory `data/lyra/split/` from the dataset repository.
- Turkish-makam, Hindustani, Carnatic
  - These datasets are part of the [CompMusic Corpora](https://compmusic.upf.edu/corpora). One should create an account to [Dunya](https://dunya.compmusic.upf.edu/) and request access to the audio files. 
  - The [pycompmusic](https://github.com/MTG/pycompmusic) has to be installed and the user `token` must be used at the helper scripts under `preprocessing/dunya/` directory. In order to fetch both data and metadata, set the `dataset` option properly (one of `'makam', 'hindustani', 'carnatic'`) and execute the scripts such as: `python preprocessing/dunya/get_audios.py --dataset 'makam'` and `python preprocessing/dunya/get_metadata.py --dataset 'carnatic'`.

### Preprocessing

The audio (mp3) files of each dataset are expected to be found under `audios/` directory at a specific path such as `data/{dataset}/`. In order to create the `mel-spectrograms` for all datasets except Lyra (for which they can be readily downloaded), use the following command by setting properly the `dataset` (use one of `'magnatagatune', 'fma', 'makam', 'hindustani', 'carnatic'`) and `data_dir` options, such as:

```bash
python preprocessing/create_mel_spectrograms.py --dataset 'magnatagatune' --data_dir '/__path_to__/magnatagatune'
```

### Splits

All splits follow the analogies 0.7, 0.1, 0.2 for training, validation and test set respectively 

- MagnaTagATune: publicly available split in training, validation and test sets, that can be found at [minzwon/sota-music-tagging-models](https://github.com/minzwon/sota-music-tagging-models/tree/master/split/mtat) repository.
- Lyra: [publicly available split](https://github.com/pxaris/lyra-dataset/tree/main/data/split) for training and test sets. The validation set is randomly splitted from the training set.
- For the rest of the datasets, i.e. FMA-medium, Turkish-makam, Hindustani and Carnatic, the splits were randomly created and they can be found at the files `train.npy`, `valid.npy` and `test.npy` under each `data/{dataset}/split` directory. Each one of thoe files contains a list of ids that is used, in turn, by the respective dataloader. 

