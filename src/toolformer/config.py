from dataclasses import dataclass

import torch


@dataclass
class ToolformerConfig:
    # General
    model_name = "EleutherAI/gpt-neo-125M"
    causal_model = True
    target_device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    # Sampling
    sampler = 'basic'

    # Inference
    max_new_tokens = 128

    # Training
    mlm_prob = 0.15
    max_length = 256
    output_path = '..'
    output_name = 'model'
    learning_rate = 1e-4
    train_batch_size = 16
    eval_batch_size = 32
    epochs = 1
    weight_decay = 0.01
    warmup_ratio = 0.1
    fp16 = False
    early_stopping_patience = 1
    test_size = 0.2

    # Filtering
    tool_call_thresh = 0