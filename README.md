# Huggingface Implementation of AV-HuBERT on the MuAViC Dataset

![lip-reading](https://github.com/facebookresearch/av_hubert/blob/main/assets/lipreading.gif)

This repository contains a Huggingface implementation of the AV-HuBERT (Audio-Visual Hidden Unit BERT) model, specifically trained and tested on the MuAViC (Multilingual Audio-Visual Corpus) dataset. AV-HuBERT is a self-supervised model designed for audio-visual speech recognition, leveraging both audio and visual modalities to achieve robust performance, especially in noisy environments.


Key features of this repository include:

- Pre-trained Models: Access pre-trained AV-HuBERT models fine-tuned on the MuAViC dataset. The pre-trained model been exported from [MuAViC](https://github.com/facebookresearch/muavic) repository.

- Inference scripts: Easily pipelines using Huggingface’s interface.

- Data preprocessing scripts: Including normalize frame rate, extract lips and audio.

### Inference code

```sh
git clone https://github.com/nguyenvulebinh/AV-HuBERT-S2S.git
cd AV-HuBERT-S2S
conda create -n avhuberts2s python=3.9
conda activate avhuberts2s
pip install -r requirements.txt
python run_example.py
```

```python
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature
from transformers import Speech2TextTokenizer
import torch

if __name__ == "__main__":
    # Load pretrained english model
    model = AV2TextForConditionalGeneration.from_pretrained('nguyenvulebinh/AV-HuBERT')
    tokenizer = Speech2TextTokenizer.from_pretrained('nguyenvulebinh/AV-HuBERT')

    # cuda
    model = model.cuda().eval()
    
    # Load normalized input data
    sample = load_feature(
        './example/lip_movement.mp4',
        "./example/noisy_audio.wav"
    )
    
    # cuda
    audio_feats = sample['audio_source'].cuda()
    video_feats = sample['video_source'].cuda()
    attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)).fill_(False).cuda()
    
    # Generate output sequence using HF interface
    output = model.generate(
        audio_feats,
        attention_mask=attention_mask,
        video=video_feats,
    )

    # decode output sequence
    print(tokenizer.batch_decode(output, skip_special_tokens=True))
    
    # check output
    assert output.detach().cpu().numpy().tolist() == [[  2,  16, 130, 516,   8, 339, 541, 808, 210, 195, 541,  79, 130, 317, 269,   4,   2]]
    print("Example run successfully")
```

### Data preprocessing scripts

```sh
mkdir model-bin
cd model-bin
wget https://huggingface.co/nguyenvulebinh/AV-HuBERT/resolve/main/20words_mean_face.npy .
wget https://huggingface.co/nguyenvulebinh/AV-HuBERT/resolve/main/shape_predictor_68_face_landmarks.dat .

# raw video only support 4:3 ratio now
cp raw_video.mp4 ./example/ 

python src/dataset/video_to_audio_lips.py
```

### Pretrained model

<table align="center">
    <tr>
        <th>Task</th>
        <th>Languages</th>
        <th>Huggingface</th>
    </tr>
    <tr>
        <td>AVSR</td>
        <th>English</th>
        <th><a href="nguyenvulebinh/AV-HuBERT">Chekpoint</a></th>
    </tr>
</table>


## Acknowledgments

**AV-HuBERT**: A significant portion of the codebase in this repository has been adapted from the original AV-HuBERT implementation.

**MuAViC Repository**: We also gratefully acknowledge the creators of the MuAViC dataset and repository for providing the pre-trained models used in this project

## License

CC-BY-NC 4.0

## Citation

```bibtex
@article{anwar2023muavic,
  title={MuAViC: A Multilingual Audio-Visual Corpus for Robust Speech Recognition and Robust Speech-to-Text Translation},
  author={Anwar, Mohamed and Shi, Bowen and Goswami, Vedanuj and Hsu, Wei-Ning and Pino, Juan and Wang, Changhan},
  journal={arXiv preprint arXiv:2303.00628},
  year={2023}
}
```