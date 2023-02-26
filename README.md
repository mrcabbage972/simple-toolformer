# simple-toolformer
# Introduction
A Python implementation of [Toolformer](https://arxiv.org/abs/2302.04761) using Huggingface Transformers

This implementation is under active development and wasn't yet verified to work end-to-end. 
Therefore, it's currently intended to be used for educational purposes only.

The immediate goal is to fine-tune a model on a downstream task and verify that this generates a lift vs fine-tuning
just the backbone on the same task.   

# Usage
First, please install the requirements file.

The example training script is at `src/scripts/train_gsm8k.py`. This would train the model on the [GSM8k](https://huggingface.co/datasets/gsm8k) dataset of Math Word Problems. 

# Contributing
If you wish to contribute to this project, please check out the existing issues or open a new one.

# Citations

```bibtex
@inproceedings{Schick2023ToolformerLM,
    title   = {Toolformer: Language Models Can Teach Themselves to Use Tools},
    author  = {Timo Schick and Jane Dwivedi-Yu and Roberto Dessi and Roberta Raileanu and Maria Lomeli and Luke Zettlemoyer and Nicola Cancedda and Thomas Scialom},
    year    = {2023}
}
```