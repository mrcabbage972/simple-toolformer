import logging
import os
from typing import List, Mapping

import torch
from datasets import Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, DataCollatorWithPadding, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    EarlyStoppingCallback, T5ForConditionalGeneration

from toolformer.config import ToolformerConfig
from toolformer.sequence_scoring import get_scores_for_labels
from toolformer.tool import Tool

logger = logging.getLogger(__name__)

class Toolformer:
    def __init__(self):
        self.cfg = ToolformerConfig()
        self.tokenizer = T5Tokenizer.from_pretrained(self.cfg.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.cfg.model_name)

    def fit(self, dataset: Dataset, tools: List[Tool]):
        """
            This is the main method for implementing the training process described in the Toolformer paper.
            It contains three stages:
            1. Use the model to generate samples of tool usage, using few-shot prompting
            2. Filter the samples by a measure of how much the improve the likelihood of the text after the tool call.
            3. Fit the model on the filtered samples.
        :param dataset:
            The dataset to fit on.
        :param tools:
            A list of tools to be considered.
        """
        logger.info('Fitting with a dataset of size {} and {} tools'.format(len(dataset), len(tools)))
        samples_for_tuning = []
        for tool in tools:
            maybe_tool_samples = self.sample_dataset(dataset, tool)
            logger.info('Examples of {} tool generation results: {}'.format(tool.get_tool_name(), ','.join(maybe_tool_samples[:2]['text'])))
            tool_samples = maybe_tool_samples.filter(lambda x: tool.text_has_call(x['text']))
            logger.info('{} samples left after filtering for tool name'.format(len(tool_samples)))
            logger.info('Examples of {} tool filtered annotations: {}'.format(tool.get_tool_name(), ','.join(maybe_tool_samples[:2]['text'])))
            executed_tool_samples = tool_samples.map(lambda x: self.execute_tool_call(x, tool))
            likely_samples = self.filter_likelihood(executed_tool_samples, tool)
            logger.info('{} samples left after filtering by likelihood'.format(len(likely_samples)))
            samples_for_tuning.append(likely_samples)
        dataset_for_tuning = concatenate_datasets(samples_for_tuning)
        if len(dataset_for_tuning) == 0:
            raise ValueError("Can't proceed: There is no data to fine-tune on!")
        self.fine_tune(dataset_for_tuning)

    def sample_dataset(self, dataset: Dataset, tool: Tool) -> Dataset:
        """
            This methods samples a dataset to produce example API calls.
            The sampling procedure is significantly simplified from what is described in the paper and likely
            wouldn't work well.
        :param dataset:
            The input texts
        :param tool:
            The tool to annotate the input texts with
        :return:
            A Dataset containing a text field and a score field.
        """
        logger.info('Sampling dataset')

        encoded_dataset = dataset.map(lambda x: self.tokenizer([tool.get_prompt_template().format(z) for z in x],
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
                                                  max_length=self.cfg.max_length,
                                                  return_dict_in_generate=True,
                                                  output_scores=True)

                all_preds += [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch_preds['sequences']]

        return Dataset.from_dict({'text': all_preds})

    def filter_likelihood(self, inputs: Dataset, tool: Tool) -> Dataset:
        logger.info('Filtering generated samples by their likelihood')

        inputs = inputs.map(lambda x: {**x,
                                       'text_before': tool.get_text_before_call(x['text']),
                                       'tool_call': tool.get_call_from_text(x['text']),
                                       'text_after': tool.get_text_after_call(x['text'])})

        inputs = inputs.map(lambda x: {**x,
                                       'tool_call_text_before': x['tool_call'] + x['text_before'],
                                       'tool_call_result_text_before': x['tool_call'] + x['tool_result'] + x[
                                           'text_before'],
                                       })

        inputs = inputs.map(lambda x: {**x,
                                       'loss_no_tool':
                                           get_scores_for_labels(x['text_before'], x['text_after'], self.model,
                                                                 self.tokenizer)[0],
                                       'loss_tool':
                                           get_scores_for_labels(inputs['tool_call_text_before'], inputs['text_after'],
                                                                 self.model, self.tokenizer)[0],
                                       'loss_tool_no_result':
                                           get_scores_for_labels(inputs['tool_call_text_before'], inputs['text_after'],
                                                                 self.model, self.tokenizer)[0]
                                       }, batched=True)

        # loss (with prefix of tool call and result ) < min(loss (with prefix of tool call), loss(no tool call)

        return inputs.filter(
            lambda x: min(x['loss_no_tool'], x['loss_tool_no_result']) - x['loss_tool'] >= self.cfg.tool_call_thresh)

    def fine_tune(self, api_call_samples: Dataset):
        """
            This is just standard HF fine-tuning.
        :param api_call_samples:
        """
        logger.info('Fine-tuning the model on {} API call samples'.format(len(api_call_samples)))
        datasets = api_call_samples.train_test_split(test_size=self.cfg.test_size)

        train_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(self.cfg.output_path, self.cfg.output_name),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=self.cfg.learning_rate,
            per_device_train_batch_size=self.cfg.train_batch_size,
            per_device_eval_batch_size=self.cfg.eval_batch_size,
            num_train_epochs=self.cfg.epochs,
            weight_decay=self.cfg.weight_decay,
            warmup_ratio=self.cfg.warmup_ratio,
            load_best_model_at_end=True,
            save_total_limit=1,
            fp16=self.cfg.fp16,
        )

        trainer = Seq2SeqTrainer(
            self.model,
            train_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            tokenizer=self.tokenizer,
            compute_metrics=None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.cfg.early_stopping_patience)]
        )

        trainer.train()

    def execute_tool_call(self, sample, tool: Tool) -> dict:
        sample['tool_result'] = tool.run(sample['text'])
        return sample
