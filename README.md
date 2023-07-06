# Cross-cultural Music Transfer learning

PyTorch implementation of cross-cultural music transfer learning using auto-tagging models.

## Reference

**From West to East: Who can understand the music of the others better?**, ISMIR 2023.

-- Charilaos Papaioannou, Emmanouil Benetos, and Alexandros Potamianos

## Datasets
- [**MagnaTagATune**](https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset)
- [**FMA**](https://github.com/mdeff/fma)
- [**Lyra**](https://github.com/pxaris/lyra-dataset)
- [**Turkish-makam**](https://dunya.compmusic.upf.edu/makam/)
- [**Hindustani**](https://dunya.compmusic.upf.edu/hindustani/)
- [**Carnatic**](https://dunya.compmusic.upf.edu/carnatic/)

## Models
- **VGG-ish** : CNN architecture, with multiple layers, that is based on the VGG network, as implemented by [Won et al.](https://arxiv.org/abs/2006.00751)
- **Musicnn** : Music inspired model that uses convolutional layers at its core, [Pons et al.](https://arxiv.org/abs/1711.02520)
- **Audio Spectrogram Transformer** : Purely attention-based model for audio classification, [Gong et al.](https://arxiv.org/abs/2104.01778)

