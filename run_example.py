import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature
from transformers import Speech2TextTokenizer
import torch

if __name__ == "__main__":
    
    model = AV2TextForConditionalGeneration.from_pretrained('nguyenvulebinh/AV-HuBERT', cache_dir='./model-bin')
    tokenizer = Speech2TextTokenizer.from_pretrained('nguyenvulebinh/AV-HuBERT', cache_dir='./model-bin')
        
    model = model.cuda().eval()
    
    sample = load_feature(
        './example/lip_movement.mp4',
        "./example/noisy_audio.wav"
    )
    
    audio_feats = sample['audio_source'].cuda()
    video_feats = sample['video_source'].cuda()
    attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)).fill_(False).cuda()
    
    output = model.generate(
        audio_feats,
        attention_mask=attention_mask,
        video=video_feats,
    )

    print(tokenizer.batch_decode(output, skip_special_tokens=True))
    assert output.detach().cpu().numpy().tolist() == [[  2,  16, 130, 516,   8, 339, 541, 808, 210, 195, 541,  79, 130, 317, 269,   4,   2]]
    print("Example run successfully")