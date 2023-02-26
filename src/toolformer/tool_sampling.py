import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import torch.nn.functional as F
from toolformer.tool import Tool

def prepare_dataset_for_sampling(dataset, tokenizer, tool):
    prompts_dataset = dataset.map(lambda x: {'prompt': tool.get_prompt_template().format(x['label'])})
    encoded_dataset = prompts_dataset.map(lambda x: tokenizer(x['prompt'],
                                                                   truncation=True, padding=True), batched=True)
    encoded_dataset.set_format(columns=['input_ids', 'attention_mask'], type='torch')
    return encoded_dataset

def postprocess_samples(all_preds, dataset, tool, is_causal_model):
    pred_ds = Dataset.from_dict({'text': all_preds,
                                 'prompt': [tool.get_prompt_template().format(z['label']) for z in dataset]})
    # prompt_end_idx = len(tool.get_prompt_template().replace('{}', '').rstrip())
    if is_causal_model:
        return pred_ds.map(lambda x: {'text': x['text'][len(x['prompt']):]})
    else:
        return pred_ds


class BasicToolSampler:
    """
    A basic tool sampler that just calls the generate method of them model

    """
    def __init__(self, tokenizer, model, cfg):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

    def sample(self, dataset: Dataset, tool: Tool) -> Dataset:
        encoded_dataset = prepare_dataset_for_sampling(dataset, self.tokenizer, tool)
        data_loader = DataLoader(encoded_dataset, batch_size=32,
                   collate_fn=DataCollatorWithPadding(self.tokenizer))
        data_iter = iter(data_loader)

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
        return postprocess_samples(all_preds, dataset, tool, self.cfg.causal_model)


class TwoStepToolSampler:
    """
    WORK IN PROGRESS

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
        self.tool_call_token_id = tokenizer.convert_tokens_to_ids(Tool.API_CALL_PREFIX)
        self.tool_call_end_token_id = tokenizer.convert_tokens_to_ids(Tool.API_CALL_SUFFIX)

    def sample(self, dataset: Dataset, tool: Tool) -> Dataset:
        encoded_dataset = prepare_dataset_for_sampling(dataset, self.tokenizer, tool)

        topk_pos_idx = self.get_topk_pos_idx(encoded_dataset, tool)
        anns_at_pos = [self.get_anns_at_pos(dataset, encoded_dataset, topk_pos_idx[:, idx], tool) for idx in range(self.top_k)]
        return concatenate_datasets(anns_at_pos)

    def get_topk_pos_idx(self, encoded_dataset, tool):
        data_loader = DataLoader(encoded_dataset, batch_size=32,
                                 collate_fn=DataCollatorWithPadding(self.tokenizer))
        data_iter = iter(data_loader)

        # TODO: create mask to cancel all tokens from the prompt template

        all_preds = []
        for inputs in data_iter:
            inputs = {k: v.to(self.cfg.target_device) for k, v in inputs.items()}
            out = self.model(**inputs)
            api_prob_at_idx = out.logits[:, :, self.tool_call_token_id]
            api_prob_at_idx[~inputs['attention_mask']] = -100
            api_prob_topk_idx = api_prob_at_idx.topk(self.top_k).indices
            all_preds.append(api_prob_topk_idx.detach())
        return torch.concat(all_preds, 0)

    def get_anns_at_pos(self, dataset, encoded_dataset, pos_idx, tool):
        # TODO: refactor to avoid having to pass two dataset objects
        # Get the text before the desired position and add the tool call token
        dataset_at_idx = encoded_dataset.add_column('pos_idx', pos_idx.numpy())\
            .map(lambda x: {'input_ids': torch.cat([x['input_ids'][:x['pos_idx']], pos_idx], -1),
                            'input_ids_suffix': x['input_ids'][x['pos_idx']:],
                            'attention_mask': torch.cat([x['attention_mask'][:x['pos_idx']], torch.ones(x['input_ids'].shape[0])], -1)})
        data_loader = DataLoader(dataset_at_idx, batch_size=32,
                                 collate_fn=DataCollatorWithPadding(self.tokenizer))
        data_iter = iter(data_loader)

        all_preds = []
        for inputs in data_iter:
            inputs = {k: v.to(self.cfg.target_device) for k, v in inputs.items()}
            batch_preds = self.model.generate(**inputs,
                                                  max_new_tokens=self.cfg.max_new_tokens,
                                                  return_dict_in_generate=True,
                                                  output_scores=True,
                                                  eos_token_id=self.tool_call_end_token_id)
            all_preds += [self.tokenizer.decode(x + y, skip_special_tokens=True) for x, y
                          in zip(batch_preds['sequences'], inputs['input_ids_suffix'])]

        return postprocess_samples(all_preds, dataset, tool)