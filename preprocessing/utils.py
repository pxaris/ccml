import os
import glob
import librosa
import tqdm
import torch
import torchaudio
import numpy as np
import librosa.display


def convert_audio_to_mel_spectrogram(audio_file, mel_specs_path, config):
    filename = os.path.basename(audio_file).split('.')[0]
    output_file = os.path.join(mel_specs_path, f'{filename}.npy')
    if os.path.isfile(output_file):
        print('File already exists - skipping.')
        return

    y, sr = librosa.load(
        audio_file, sr=config['audio_sr'], duration=config['max_audio_duration'])

    spectrogram = config['operators']['torch_specgram'](
        torch.from_numpy(y))
    spectrogram_dB = config['operators']['torch_to_db'](spectrogram)

    np.save(output_file, spectrogram_dB)


def create_mel_spectrograms(config, mel_specs_dir):
    config['operators'] = {'torch_specgram': torchaudio.transforms.MelSpectrogram(sample_rate=config['audio_sr'],
                                                                                  n_fft=config['n_fft'],
                                                                                  f_min=config['f_min'],
                                                                                  f_max=config['f_max'],
                                                                                  n_mels=config['n_mels']),
                           'torch_to_db': torchaudio.transforms.AmplitudeToDB()
                           }
    audios_dir = os.path.join(config['data_dir'], 'audios')
    audios = glob.glob(f'{audios_dir}/**/*.mp3', recursive=True)
    
    for audio in tqdm.tqdm(audios):
        try:
            convert_audio_to_mel_spectrogram(audio, mel_specs_dir, config)
        except:
            print(f'An error occurred; file "{audio}" was not converted.')
    print(f'Done. Files are under: {mel_specs_dir}')
