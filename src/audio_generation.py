from TTS.utils.io import load_config
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.io import load_checkpoint
from TTS.vocoder.utils.generic_utils import setup_generator

import sys
import os
import torch
import time
import IPython

# for some reason TTS installation does not work on Colab
# sys.path.append('TTS_repo')

with open('config/tts.json', 'r') as f:
    config = json.load(f)

def interpolate_vocoder_input(scale_factor, spec):
  """Interpolation to tolarate the sampling rate difference
  btw tts model and vocoder"""
  print(" > before interpolation :", spec.shape)
  spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)
  spec = torch.nn.functional.interpolate(spec, scale_factor=scale_factor, mode='bilinear').squeeze(0)
  print(" > after interpolation :", spec.shape)
  return spec


def tts(model, text, CONFIG, use_cuda, ap, use_gl, figures=True):
  t_1 = time.time()
  # run tts
  target_sr = CONFIG.audio['sample_rate']
  waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs =\
    synthesis(model,
              text,
              CONFIG,
              use_cuda,
              ap,
              speaker_id,
              None,
              False,
              CONFIG.enable_eos_bos_chars,
              use_gl)
  # run vocoder
  mel_postnet_spec = ap._denormalize(mel_postnet_spec.T).T
  if not use_gl:
      target_sr = VOCODER_CONFIG.audio['sample_rate']
      vocoder_input = ap_vocoder._normalize(mel_postnet_spec.T)
      if scale_factor[1] != 1:
          vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
      else:
          vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
      waveform = vocoder_model.inference(vocoder_input)
  # format output
  if use_cuda and not use_gl:
      waveform = waveform.cpu()
  if not use_gl:
      waveform = waveform.numpy()
  waveform = waveform.squeeze()
  # compute run-time performance
  rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
  tps = (time.time() - t_1) / len(waveform)
  print(waveform.shape)
  print(" > Run-time: {}".format(time.time() - t_1))
  print(" > Real-time factor: {}".format(rtf))
  print(" > Time per step: {}".format(tps))
  # display audio
  IPython.display.display(IPython.display.Audio(waveform, rate=target_sr))  
  return alignment, mel_postnet_spec, stop_tokens, waveform


# runtime settings
use_cuda = False

# model paths
TTS_MODEL = config.get("TTS_MODEL")
TTS_CONFIG = config.get("TTS_CONFIG")
VOCODER_MODEL = config.get("VOCODER_MODEL")
VOCODER_CONFIG = config.get("VOCODER_CONFIG")

# load configs
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

VOCODER_CONFIG.audio['stats_path'] = config.get("STATS_PATH")

def generate_audio(text):

    # load the audio processor
    ap = AudioProcessor(**TTS_CONFIG.audio)   

    # LOAD TTS MODEL
    # multi speaker 
    speakers = []
    speaker_id = None
        
    #if 'characters' in TTS_CONFIG.keys():
    #    symbols, phonemes = make_symbols(**c.characters)

    # load the model
    num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
    model = setup_model(num_chars, len(speakers), TTS_CONFIG)      

    # load model state
    model, _ =  load_checkpoint(model, TTS_MODEL, use_cuda=use_cuda)
    model.eval();
    model.store_inverse();

    # LOAD VOCODER MODEL
    vocoder_model = setup_generator(VOCODER_CONFIG)
    vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
    vocoder_model.remove_weight_norm()
    vocoder_model.inference_padding = 0

    # scale factor for sampling rate difference
    scale_factor = [1,  VOCODER_CONFIG['audio']['sample_rate'] / ap.sample_rate]
    print(f"scale_factor: {scale_factor}")

    ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])    
    if use_cuda:
        vocoder_model.cuda()
    vocoder_model.eval();

    model.length_scale = config.get("LENGTH_SCALE") # set speed of the speech. 
    model.noise_scale = config.get("NOISE_SCALE")  # set speech variationd

    # gen_text =  "Bill got in the habit of asking himself “Is that thought true?” and if he wasn’t absolutely certain it was, he just let it go."
    align, spec, stop_tokens, wav = tts(model, text, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True)
    
    return align, spec, stop_tokens, wav