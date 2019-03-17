import importlib.resources

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential

import keras.utils as ku
import pathlib
import numpy as np


BIBLE_CORPUS = importlib.resources.path('religious_gan.corpus',
                                        'bible.txt')


class LSTMNetwork:
    def __init__(self,):
        self.corpuse: str = None

    def load_corpus(self, corpus_path: pathlib.Path) -> str:
        corpus_path = pathlib.Path(corpus_path)
        with corpus_path.open() as corpus_file:
            corpus = corpus_file.read()
        return corpus

    def dataset_preperation(self,):
        ...

    def create_model(self,):
        ...

    def generate_text(self):
        ...


if __name__ == "__main__":
    corpus_path = next(BIBLE_CORPUS.gen)
