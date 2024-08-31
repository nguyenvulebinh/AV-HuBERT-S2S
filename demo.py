import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gradio as gr
from src.dataset.video_to_audio_lips import process_raw_data_for_avsr
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature
from transformers import Speech2TextTokenizer
import torch
import time

model = AV2TextForConditionalGeneration.from_pretrained('nguyenvulebinh/AV-HuBERT', cache_dir='./model-bin')
tokenizer = Speech2TextTokenizer.from_pretrained('nguyenvulebinh/AV-HuBERT', cache_dir='./model-bin')

if torch.cuda.is_available():
    model = model.cuda().eval()
else:
    model = model.eval()

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

def video_identity(video):
    start_time = time.time()
    output = process_raw_data_for_avsr(video)
    process_raw_data_for_avsr_process_time = time.time() - start_time
    print(f"Time taken to process video: {process_raw_data_for_avsr_process_time:.2f}s")
    print("output process_raw_data_for_avsr", output)
    start_time = time.time()
    text_output = infer_avsr(output['audio'], output['lip_movement'])
    text_output_process_time = time.time() - start_time
    print(f"Time taken to infer AVSR: {text_output_process_time:.2f}s")
    return output['lip_video_path'], f"Process video time: {process_raw_data_for_avsr_process_time:.2f}s\nInfer audio visual time: {text_output_process_time:.2f}s", text_output

demo = gr.Interface(video_identity,
                    gr.Video(),
                    [
                        gr.Video(label="Lip Movement", include_audio=True, height=256, width=256),
                        gr.Text(label="Process time"),
                        gr.Text(label="Text output"),
                    ]
                    )

if __name__ == "__main__":
    demo.launch(debug=True)