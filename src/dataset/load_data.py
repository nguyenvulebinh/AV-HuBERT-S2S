import numpy as np
from python_speech_features import logfbank
from scipy.io import wavfile
import torch
import torch.nn.functional as F
from .utils import load_video, Compose, Normalize, CenterCrop, load_video

def load_feature(video_path, audio_path):
    """
    Load image and audio feature
    Returns:
    video_feats: numpy.ndarray of shape [T, H, W, 1], audio_feats: numpy.ndarray of shape [T, F]
    """
    def stacker(feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats
    # video_fn, audio_fn = mix_name
    # if 'video' in self.modalities:
    video_feats = load_video_features(video_path) # [T, H, W, 1]
    # else:
        # video_feats = None
    # if 'audio' in self.modalities:
    # audio_fn = audio_fn.split(':')[0]
    sample_rate, wav_data = wavfile.read(audio_path)
    assert sample_rate == 16_000 and len(wav_data.shape) == 1
    
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32) # [T, F]
    audio_feats = stacker(audio_feats, 4) # [T/stack_order_audio, F*stack_order_audio]
    # else:
    #     audio_feats = None
    if audio_feats is not None and video_feats is not None:
        diff = len(audio_feats) - len(video_feats)
        if diff < 0:
            audio_feats = np.concatenate([audio_feats, np.zeros([-diff, audio_feats.shape[-1]], dtype=audio_feats.dtype)])
        elif diff > 0:
            audio_feats = audio_feats[:-diff]


    audio_feats, video_feats = torch.from_numpy(audio_feats.astype(np.float32)) if audio_feats is not None else None, torch.from_numpy(video_feats.astype(np.float32)) if video_feats is not None else None
    # if self.normalize and 'audio' in self.modalities:
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])

    # audio_feats shape [batch, F, T]
    audio_feats = audio_feats.permute(1, 0).unsqueeze(0)

    # video_feats shape [batch, C, T, H, W]
    video_feats = video_feats.permute(3, 0, 1, 2).unsqueeze(0)

    return {"video_source": video_feats, 'audio_source': audio_feats}

def load_video_features(video_path):

    image_crop_size = 88
    image_mean = 0.421
    image_std = 0.165
    transform = Compose([
        Normalize( 0.0,255.0 ),
        CenterCrop((image_crop_size, image_crop_size)),
        Normalize(image_mean, image_std) 
    ])

    feats = load_video(video_path)
    feats = transform(feats)
    feats = np.expand_dims(feats, axis=-1)
    return feats

