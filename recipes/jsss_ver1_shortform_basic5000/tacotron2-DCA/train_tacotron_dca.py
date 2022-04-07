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


# from TTS.tts.datasets.tokenizer import Tokenizer

#output_path = os.path.dirname(os.path.abspath(__file__))
output_path = "../jsss_ver1_shortform_basic5000-1.1/"
print(output_path)

# init configs
dataset_config = BaseDatasetConfig(
    name="jsss_ver1_shortform_basic5000", 
    path = "/home/nidhi/Documents/animedub/audio/data/jsss_ver1/short-form/basic5000",
    meta_file_train="/home/nidhi/Documents/animedub/audio/data/jsss_ver1/short-form/basic5000/metadata.txt", 
    language= "ja",
    #path=os.path.join(output_path, "../jsss_ver1_shortform_basic5000-1.1/")
    )

from TTS.tts.datasets import load_tts_samples
# custom formatter implementation
def formatter(root_path, manifest_file, **kwargs):  # pylint: disable=unused-argument
    """Assumes each line as ```<filename>|<transcription>```
    """
    txt_file = os.path.join(root_path, manifest_file)
    items = []
    speaker_name = "my_speaker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            print("________________________________________________________________________")
            cols = line.split("\t")
            print(cols)
            wav_file = os.path.join(root_path, "wavs", cols[0])
            print(wav_file)
            text = cols[1]
            items.append({"text":text, "audio_file":wav_file, "speaker_name":speaker_name})
            print("append_complete")
    return items

#print(wav_file)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)



config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    ga_alpha=0.0,
    decoder_loss_alpha=0.25,
    postnet_loss_alpha=0.25,
    postnet_diff_spec_alpha=0,
    decoder_diff_spec_alpha=0,
    decoder_ssim_alpha=0,
    postnet_ssim_alpha=0,
    r=2,
    attention_type="dynamic_convolution",
    double_decoder_consistency=False,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)


# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
#train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)
# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True, formatter=formatter)

print("______________________________________Train_samples[0]________________________________________________")
print(train_samples[0])

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = Tacotron2(config, ap, tokenizer)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
     TrainerArgs(),config, output_path = output_path, model=model, train_samples= train_samples, eval_samples= eval_samples)

# AND... 3,2,1... 🚀
trainer.fit()




















