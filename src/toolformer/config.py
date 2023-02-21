from dataclasses import dataclass


@dataclass
class ToolformerConfig:
    # General
    model_name = "google/flan-t5-small"
    target_device = 'cpu'

    # Training
    max_length = 64
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