from typing import List, Tuple

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Dense, Dropout
from keras.layers import LSTM as KLSTM
from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
from keras.models import Sequential

import keras.utils as ku
import keras

import pathlib
import numpy as np


class LSTM:
    def __init__(self, corpus_path):
        self.corpus = self.load_corpus(corpus_path)
        self.tokenizer = Tokenizer()
        self.model: keras.engine.sequential.Sequential = None
        self.predictors: List[List[str]] = None
        self.label: List[str] = None
        self.max_sequence_len: int = None
        self.total_words: int = None

    def load_corpus(self, corpus_path: pathlib.Path) -> str:
        corpus_path = pathlib.Path(corpus_path)
        with corpus_path.open() as corpus_file:
            corpus = corpus_file.read()
        return corpus

    def dataset_preperation(self,) -> Tuple[List[List[str]], List[str],
                                            int, int]:
        # first tokenize
        corpus = self.corpus.lower().split("\n")
        self.tokenizer.fit_on_texts(corpus)
        total_words = len(self.tokenizer.word_index) + 1

        # second convert the corpus into a flat dataset of sentence sequences
        input_sequences = list()
        for line in corpus:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        # third ensure all sequenes are equal in length
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(
                pad_sequences(input_sequences,
                              maxlen=max_sequence_len,
                              padding='pre'))

        # next we build the n-gram sequence for predictor/label model
        predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
        label = ku.to_categorical(label, num_classes=total_words)
        self.predictors = predictors
        self.label = label
        self.max_sequence_len = max_sequence_len
        self.total_words = total_words
        return predictors, label, max_sequence_len, total_words

    def create_model(self,) -> keras.engine.sequential.Sequential:
        input_len = self.max_sequence_len - 1
        model = Sequential()
        model.add(Embedding(self.total_words, 10, input_length=input_len))
        model.add(KLSTM(150))
        model.add(Dropout(0.1))
        model.add(Dense(self.total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.fit(self.predictors, self.label, epochs=100, verbose=1)
        self.model = model
        model_json = model.to_json()
        json_path = 'model/model.json'
        weights_path = 'model/model_weights.h5'
        model_path = 'model/model.h5'

        with open(str(json_path), 'w') as json_file:
            json_file.write(model_json)
        model.save_weights(str(weights_path))
        model.save(str(model_path))
        other_data = 'model/model_data.txt'
        with open(other_data, 'w') as data_file:
            data_file.write(f"total_words: {self.total_words}\n")
            data_file.write(f"max_sequence_len: {self.max_sequence_len}")

    def generate_text(self,
                      seed_text: str,
                      next_words: int,):
        for j in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list],
                                       maxlen=self.max_sequence_len - 1,
                                       padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=1)

            output_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            seed_text += " " + output_word
        return seed_text


if __name__ == "__main__":
    corpus_path = pathlib.Path('corpus/bible.txt')
    lstm = LSTM(corpus_path)
    lstm.dataset_preperation()
    lstm.create_model()
    text = lstm.generate_text("God and", 20)
    print(text)
    text = lstm.generate_text("God is false", 20)
    print(text)
    text = lstm.generate_text("Who is God", 20)
    print(text)
    text = lstm.generate_text("The real God is", 20)
    print(text)
