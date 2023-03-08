import logging
import re

from datasets import load_dataset

from toolformer.toolformer import Toolformer
from toolformer.tools.calculator import CalculatorTool
from toolformer.tools.date import DateTool

logging.basicConfig(level=logging.INFO)

def main():
    tf = Toolformer()

    dataset = load_dataset("gsm8k", 'main', split="train").select(range(5))
    dataset = dataset.rename_column('question', 'input')
    dataset = dataset.rename_column('answer', 'label')
    dataset = dataset.map(lambda x: {'input': x['input'],
                                     'label': re.sub("(<<).*?(>>)", "", x['label']).split('####')[0]
                          .rstrip().replace('\n', ' ')})
    apis = [CalculatorTool()]

    tf.fit(dataset, apis)


if __name__ == '__main__':
    main()