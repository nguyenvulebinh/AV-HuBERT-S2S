import torch

def convert_checkpoint_to_encoder_state_dict(ckpt_path):
        # load state dict from checkpoint
    seq2seq_checkpoint = torch.load(ckpt_path)
    encoder_checkpoint = dict()
    for key in list(seq2seq_checkpoint['model'].keys()):
        if "encoder.w2v_model" in key:
            encoder_checkpoint[key.replace("encoder.w2v_model.", "")] = seq2seq_checkpoint['model'][key]
    
    MAPPING_FAIRSEQ_TO_HF = {
        "pos_conv.0": "pos_conv_embed.conv",
        "self_attn.k_proj": "attention.k_proj",
        "self_attn.v_proj": "attention.v_proj",
        "self_attn.q_proj": "attention.q_proj",
        "self_attn.out_proj": "attention.out_proj",
        "self_attn_layer_norm": "layer_norm",
        "fc1": "feed_forward.intermediate_dense",
        "fc2": "feed_forward.output_dense",
        "final_layer_norm": "final_layer_norm",
    }
    
    # encoder_checkpoint_hf = dict()
    old_keys = list(encoder_checkpoint.keys())
    for key in old_keys:
        for k, v in MAPPING_FAIRSEQ_TO_HF.items():
            if k in key:
                encoder_checkpoint[key.replace(k, v)] = encoder_checkpoint.pop(key)
    return encoder_checkpoint

def convert_checkpoint_to_decoder_state_dict(ckpt_path):
    seq2seq_checkpoint = torch.load(ckpt_path)
    decoder_checkpoint = dict()
    for key in list(seq2seq_checkpoint['model'].keys()):
        if key.startswith("decoder."):
            new_key = key.replace("decoder.", "")
            decoder_checkpoint[new_key] = seq2seq_checkpoint['model'][key]
    return decoder_checkpoint