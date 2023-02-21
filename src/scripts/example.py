import logging

from datasets import load_dataset

from toolformer.toolformer import Toolformer
from toolformer.tools.date import DateTool

logging.basicConfig(level=logging.INFO)

def main():
    tf = Toolformer()

    # TODO: replace with a more useful dataset
    dataset = load_dataset("rotten_tomatoes", split="train").select(range(2))
    apis = [DateTool()]

    tf.fit(dataset, apis)


if __name__ == '__main__':
    main()