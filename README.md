# simple-toolformer
# Introduction
A Python implementation of [Toolformer](https://arxiv.org/abs/2302.04761) using Huggingface Transformers

This implementation is still a work-in-progress and it still wasn't tested end-to-end. 
Therefore, it's intended to be used for educational purposes only.

The main difference compared to the paper is that  I simplified the procedure for sampling API calls from unlabeled texts.
# Usage
First, please install the requirements file.

The example training script is at `src/scripts/train_gsm8k.py`. This would train the model on the [GSM8k](https://huggingface.co/datasets/gsm8k) dataset of Math Word Problems. 