import os
from trainer import Trainer, TrainerArgs
import sys
import TTS
print(os.path.dirname(TTS.__file__))
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path =  "../jsss_ver1_shortform_basic5000-1.1/"
print(output_path)

#debugging FileNotFound error on a wav file sample
filename = '/home/nidhi/Documents/animedub/audio/data/jsss_ver1/short-form/basic5000/wavs/BASIC5000_2771.wav'
print(os.stat(filename).st_size)

#debugging AssertionError in sample_rate vs sr in audio.py
import soundfile as sf
x, sr = sf.read(filename)
print(x)
print(sr)