from torch import nn
from transformers import Speech2TextModel, Speech2TextForConditionalGeneration
from transformers.models.speech_to_text.modeling_speech_to_text import Speech2TextDecoder
from transformers import Wav2Vec2Config
from .avhubert import AVHubertModel

class AV2TextModel(Speech2TextModel):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = AVHubertModel(config)
        self.decoder = Speech2TextDecoder(config)
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.decoder.embed_tokens.weight

class AV2TextForConditionalGeneration(Speech2TextForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = AV2TextModel(config)
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)
        self.lm_head.weight = self.model.decoder.embed_tokens.weight