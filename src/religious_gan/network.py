import importlib.resources

# from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Embedding, LSTM, Dense
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.models import Sequential
#
# import keras.utils as ku
import pathlib
import numpy as np


BIBLE_CORPUS = importlib.resources.path('religious_gan.corpus',
                                        'bible.txt')


def load_corpus(corpus_path: pathlib.Path) -> str:
    corpus = None
    corpus_path = pathlib.Path(corpus_path)
    with corpus_path.open() as corpus_file:
        corpus = corpus_file.readlines()
    return corpus


if __name__ == "__main__":
    corpus_path = next(BIBLE_CORPUS.gen)
    corpus = load_corpus(corpus_path)
    print(type(corpus))
