import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import gradio as gr
from src.dataset.video_to_audio_lips import process_raw_data_for_avsr
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature
from transformers import Speech2TextTokenizer
import torch
import time
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

model = AV2TextForConditionalGeneration.from_pretrained('nguyenvulebinh/AV-HuBERT', cache_dir='./model-bin')
tokenizer = Speech2TextTokenizer.from_pretrained('nguyenvulebinh/AV-HuBERT', cache_dir='./model-bin')

if torch.cuda.is_available():
    model = model.cuda().eval()
else:
    model = model.eval()

def load_noise_samples(noise_path=Path("./example/")):
    noise_dict = defaultdict(list)
    for wav_filepath in (noise_path/"noise_samples").rglob('*.wav'):
        category = wav_filepath.parent.stem
        noise_dict[category].append(str(wav_filepath))
    return noise_dict

def infer_avsr(audio_path, lip_movement_path):
    sample = load_feature(
        lip_movement_path,
        audio_path
    )

    audio_feats = sample['audio_source']
    video_feats = sample['video_source']
    attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)).fill_(False)

    if torch.cuda.is_available():
        audio_feats = audio_feats.cuda()
        video_feats = video_feats.cuda()
        attention_mask = attention_mask.cuda()

    output = model.generate(
        audio_feats,
        attention_mask=attention_mask,
        video=video_feats,
    )

    text_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    return text_output

def handling_video(video, noise_snr, noise_type):
    print(f"Mixing audio with `{noise_type}` noise (SNR={noise_snr}).")
    noise_wav_files = NOISE[noise_type]
    noise_wav_file = noise_wav_files[random.randint(0, len(noise_wav_files)-1)]
    # print(f"Noise Wav used is {noise_wav_file}")


    start_time = time.time()
    output = process_raw_data_for_avsr(video, noise_wav_file, noise_snr)
    process_raw_data_for_avsr_process_time = time.time() - start_time
    print(f"Time taken to process video: {process_raw_data_for_avsr_process_time:.2f}s")
    print("output process_raw_data_for_avsr", output)
    start_time = time.time()
    text_output = infer_avsr(output['audio'], output['lip_movement'])
    text_output_process_time = time.time() - start_time
    print(f"Time taken to infer AVSR: {text_output_process_time:.2f}s")
    return output['lip_video_path'], f"Process video time: {process_raw_data_for_avsr_process_time:.2f}s\nInfer audio visual time: {text_output_process_time:.2f}s", text_output

if __name__ == "__main__":
    NOISE = load_noise_samples()

    demo = gr.Interface(
        handling_video,
        [
            gr.Video(),
            gr.Slider(
                label="SNR", minimum=-20, maximum=20, step=5, value=20
            ),
            gr.Radio(
                label="Noise", choices=list(NOISE.keys()),
                value=sorted(NOISE.keys())[0]
            )
        ],
        [
            gr.Video(label="Lip Movement", include_audio=True, height=256, width=256),
            gr.Text(label="Process time"),
            gr.Text(label="Text output"),
        ]
    )

    demo.launch(debug=True)