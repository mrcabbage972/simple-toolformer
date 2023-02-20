import os
from typing import List, Mapping

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, DataCollatorWithPadding, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    EarlyStoppingCallback, T5ForConditionalGeneration

from toolformer.sequence_scoring import get_scores_for_labels
from toolformer.tool import Tool


class Toolformer:
    def __init__(self):
        self.model_name = "google/flan-t5-small"
        self.target_device = 'cpu'
        self.max_length = 64
        self.output_path = '..'
        self.output_name = 'model'
        self.learning_rate = 1e-4
        self.train_batch_size = 16
        self.eval_batch_size = 32
        self.epochs = 1
        self.weight_decay = 0.01
        self.warmup_ratio = 0.1
        self.fp16 = False
        self.early_stopping_patience = 1

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

        self.api_call_thresh = 0

    def fit(self, dataset: Dataset, tools: List[Tool]):
        samples_for_tuning = []
        for tool in tools:
            maybe_api_samples = self.sample_dataset(dataset, tool)
            api_samples = maybe_api_samples.filter(lambda x: tool.text_has_api_call(x['text']))
            executed_api_samples = api_samples.map(lambda x: self.execute_api_call(x, tool))
            likely_samples = self.filter_likelihood(executed_api_samples, tool)
            samples_for_tuning.append(likely_samples)
        self.fine_tune(likely_samples) # TODO: convert to dataset

    def sample_dataset(self, dataset: Dataset, tool: Tool) -> Dataset:
        """
            This methods samples a dataset to produce example API calls.
            The current implementation is significantly simplified from what is described in the paper and likely
            wouldn't work well.
        :param dataset:
            The input texts
        :param tool:
            The tool to annotate the input texts with
        :return:
            A Dataset containing a text field and a score field.
        """
        encoded_dataset = dataset.map(lambda x: self.tokenizer([tool.get_prompt_template().format(z) for z in x],
                                                               truncation=True, padding=True), batched=True)
        encoded_dataset.set_format(columns=['input_ids', 'attention_mask'], type='torch')
        test_data_loader = DataLoader(encoded_dataset, batch_size=32,
                                      collate_fn=DataCollatorWithPadding(self.tokenizer))
        data_iter = iter(test_data_loader)

        all_preds = []
        for inputs in data_iter:
            inputs = {k: v.to(self.target_device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_preds = self.model.generate(**inputs,
                                                  max_length=self.max_length,
                                                  return_dict_in_generate=True,
                                                  output_scores = True)

                all_preds += [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch_preds['sequences']]

        return Dataset.from_dict({'text': all_preds})

    def filter_likelihood(self, inputs: Dataset, tool: Tool) -> Dataset:
        inputs = inputs.map(lambda x: x.update({'text_before': tool.get_text_before_api_call(x['text']),
                              'api_call': tool.get_text_after_api_call(x['text']),
                              'text_after': tool.get_text_after_api_call(x['text'])}))

        loss_no_api = get_scores_for_labels(inputs['text_before'], inputs['text_after'], self.model, self.tokenizer)
        loss_api = get_scores_for_labels(inputs['api_result_text_before'], inputs['text_after'], self.model, self.tokenizer)
        loss_api_no_result = get_scores_for_labels(inputs['api_text_before'], inputs['text_after'], self.model, self.tokenizer)

        # loss (with prefix of api call and result ) < min(loss (with prefix of api call), loss(no api call)

        return inputs.filter(min(loss_no_api, loss_api_no_result) - loss_api >= self.api_call_thresh) # this line doesn't work yet

    def fine_tune(self, likely_samples: Dataset):
        datasets = likely_samples.train_test_split(test_size=0.2)

        train_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(self.output_path, self.output_name),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="pearson",
            greater_is_better=True,
            save_total_limit=1,
            fp16=self.fp16,
        )

        trainer = Seq2SeqTrainer(
            self.model,
            train_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["test"],
            tokenizer=self.tokenizer,
            compute_metrics=None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)]
        )

        trainer.train()

    def execute_api_call(self, sample, tool: Tool) -> dict:
        sample['tool_result'] = tool.run(sample['text'])
        return sample
