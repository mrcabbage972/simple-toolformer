import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from toolformer.tool import Tool


class BasicToolSampler:
    """
    A basic tool sampler that just calls the generate method of them model

    """
    def __init__(self, tokenizer, model, cfg):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def sample(self, dataset: Dataset, tool: Tool) -> Dataset:
        prompts_dataset = dataset.map(lambda x: {'prompt': tool.get_prompt_template().format(x['label'])})
        encoded_dataset = prompts_dataset.map(lambda x: self.tokenizer(x['prompt'],
                                                                       truncation=True, padding=True), batched=True)
        encoded_dataset.set_format(columns=['input_ids', 'attention_mask'], type='torch')
        test_data_loader = DataLoader(encoded_dataset, batch_size=32,
                                      collate_fn=DataCollatorWithPadding(self.tokenizer))
        data_iter = iter(test_data_loader)

        all_preds = []
        for inputs in data_iter:
            inputs = {k: v.to(self.cfg.target_device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_preds = self.model.generate(**inputs,
                                                  max_new_tokens=self.cfg.max_new_tokens,
                                                  return_dict_in_generate=True,
                                                  output_scores=True)

                all_preds += [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch_preds['sequences']]

        # This is a bit ugly due to iterating over the dataset manually
        pred_ds = Dataset.from_dict({'text': all_preds,
                                     'prompt': [tool.get_prompt_template().format(z['label']) for z in dataset]})
        # prompt_end_idx = len(tool.get_prompt_template().replace('{}', '').rstrip())
        if self.cfg.causal_model:
            return pred_ds.map(lambda x: {'text': x['text'][len(x['prompt']):]})
        else:
            return pred_ds


class TwoStepToolSampler:
    """
    Implements the sampling procedure as detailed in the paper:
     First, sample K positions for the [ token.
     Then, sample M sequences out of each of the K.
    """

    def __init__(self, tokenizer, model, cfg, top_k, num_seq_per_pos):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.num_seq_per_pos = num_seq_per_pos
        self.top_k = top_k

    def sample(self, dataset: Dataset, tool: Tool) -> Dataset:
        topk_pos_idx = self.get_topk_pos_idx(dataset, tool)
        anns_at_pos = [self.get_anns_at_pos(topk_pos_idx, idx) for idx in range(self.top_k)]
        return concatenate_datasets(anns_at_pos)

    def get_topk_pos_idx(self, dataset, tool):
        pass

    def get_anns_at_pos(self, topk_pos_idx, idx):
        pass
