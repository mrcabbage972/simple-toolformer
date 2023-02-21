import logging

from datasets import load_dataset

from toolformer.toolformer import Toolformer
from toolformer.tools.calculator import CalculatorTool
from toolformer.tools.date import DateTool

logging.basicConfig(level=logging.INFO)

def main():
    tf = Toolformer()

    dataset = load_dataset("gsm8k", split="train").select(range(2))
    apis = [DateTool(), CalculatorTool()]

    tf.fit(dataset, apis)


if __name__ == '__main__':
    main()