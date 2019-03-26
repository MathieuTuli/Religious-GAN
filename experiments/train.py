#!/usr/bin/env python3
import torch


import math
import os
import random
import re
import time

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

print(f"PyTorch Version: {torch.__version__}")
print(f"Cuda availability: {torch.cuda.is_available()}")
torch.manual_seed(11111)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def get_text_files(folder_name):
    """
    Get text files from uploaded files
    """
    text_files = []
    for file in os.listdir(folder_name):
        if file.endswith(".txt"):
            text_files.append(os.path.join(folder_name,file))
    return sorted(text_files)


def store_files(file_list):
    """
    Read and store files in the file_list as strings 
    Returns a dictionary with keys as file names and values as their contents
    """
    output = {}
    for file in file_list:
        with open(file, 'r') as f:
            output[file] = f.read()
    return output
        

def dictionary(training_data):
    """
    Create word2idx and idx2word dictionaries 
    Returns two dictionaries
    """
    word2idx = {}
    idx2word = {}
    words = training_data.split()
    token = 0
    for word in words: 
        if not word in word2idx:
            word2idx[word] = token
            idx2word[token] = word
            token +=1
    return word2idx, idx2word


def tensor_generator(training_data, word2idx):
    """
    Create a PyTorch LongTensor for the training data
    Returns a 1D tensor with ids of words  
    """
    words = training_data.split()
    ids = torch.LongTensor(len(words))
    
    for i, word in enumerate(words):
        ids[i] = word2idx[word]
        
    return ids

        
def batchify(data, bsz):
    """
    Batch the data with size bsz
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    """
    Select seq_length long batches at once for training
    Data and Targets will be differed in index by one
    """
    seq_len = min(seq_length, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    For truncated backpropagation 
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


class RNNModel(nn.Module):
    """
    Container module with an encoder, a recurrent module, and a decoder.
    PyTorch provides the facility to write custom models
    """

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.LSTM = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.LSTM(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
    

def evaluate(data_source, eval_batch_size):
    """
    Evaluates the performance of the trained model in input data source
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = vocab_size
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_length):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train():
    """
    Training - Full Throttle :D 
    """
    # Turn on training mode which enables dropout.
    
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = vocab_size
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, seq_length)):
        data, targets = get_batch(train_data, i)
        
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we don't, the model would try backpropagating all the way to start of the network.
        
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        clip_grad_norm_(model.parameters(), 0.25)
        for p in model.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        total_loss += loss.item()

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:5.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} |'.format(
                epoch, batch, len(train_data) // seq_length, learning_rate,
                elapsed * 1000 /log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
            
            
def get_warmup_state(data_source):
    """
    Starting hidden states as zeros might not deliver the context 
    So a warm up is on a desired primer text 
    Returns the hidden state for actual generation 
    """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = vocab_size
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, seq_length):
            data, targets = get_batch(data_source, i)
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)         

# #Once run this will allow you to manually select the path on local drive for file you wish to upload
# uploaded = files.upload()
# for file in uploaded.keys():
#     print('User uploaded file "{name}" with length {length} bytes'.format(name=file, length=len(uploaded[file])))
#     return hidden

txt_files = get_text_files("playground_data/")
print("Files uploaded: ", ", ".join(txt_files))
print("Number of files: ", len(txt_files))

corpora = store_files(txt_files)

# Copy complete corpora to training
training_files = list(corpora.keys())

# Select a file for validation and remove it from training data
val_file = random.choice(training_files)
training_files.pop(training_files.index(val_file))

# Select a file for test and remove it from training data
test_file = random.choice(training_files)
training_files.pop(training_files.index(test_file))

print("Training files ({}): {}".format(len(training_files), ", ".join(training_files)))
print("Validation file : ", val_file)
print("Testing files : ", test_file)

# Get the vocabulary by combining all books

combined_corpus = " ".join(list(corpora.values()))

word2idx, idx2word = dictionary(combined_corpus)

print("Total number of words in entire corpus: ", len(combined_corpus.split()))

vocab_size = len(word2idx)

print("Vocabulary size: ", vocab_size)

del combined_corpus

print("Index of Jesus: ", word2idx["Jesus"])
print("Word of index {}:  {}".format(word2idx["Jesus"], idx2word[word2idx["Jesus"]]))

# Hyper-parameters

embed_size = 300
hidden_size = 1024
num_layers = 2
num_epochs = 100
batch_size = 30
seq_length = 35
learning_rate = 20.0
dropout_value = 0.4
log_interval = 100
eval_batch_size = 10

# Concatenate the training data into a single corpus by selecting corresponding files from corpora
training_data = ""
for file in training_files:
    training_data += corpora[file] + " "
training_data = training_data.rstrip()

# Batchify every data set
train_data = batchify(tensor_generator(training_data, word2idx), batch_size)
val_data = batchify(tensor_generator(corpora[val_file], word2idx), eval_batch_size)
test_data = batchify(tensor_generator(corpora[test_file],word2idx), eval_batch_size)

print("Shape of batchified training data: ", train_data.shape)
print("Shape of batchified validation data: ", val_data.shape)
print("Shape of batchified testing data: ", test_data.shape)

# Define model for training 

model = RNNModel(ntoken=vocab_size, ninp=embed_size, nhid=hidden_size, nlayers=num_layers, dropout=dropout_value).to(device)
criterion = nn.CrossEntropyLoss()

# Start training the model

best_val_loss = None
training_loss = []
validation_loss = []

# At any point you can hit Ctrl + C to break out of training early.
try:
    for epoch in range(1, num_epochs+1):
        break
        epoch_start_time = time.time()
        
        # Start training for one epoch
        train()
        
        # Get and store validation and training losses 
        val_loss = evaluate(val_data, eval_batch_size)
        tr_loss = evaluate(train_data, batch_size)
        
        
        training_loss.append(tr_loss)
        validation_loss.append(val_loss)
        
        print('-' * 122)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | training loss {:5.2f} | training ppl {:8.2f} |'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss), tr_loss, math.exp(tr_loss)))
        print('-' * 122)
        
        
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            torch.save(model.state_dict(), "model.pt")
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            learning_rate = learning_rate /1.5
except KeyboardInterrupt:
    print('-' * 122)
    print('Exiting from training early')

# Define model and load the states
# Remember, it won't work if you've changed the hyperparameters

model = RNNModel(ntoken=vocab_size, ninp=embed_size, nhid=hidden_size, nlayers=num_layers, dropout=dropout_value).to(device)
model.load_state_dict(torch.load("model.pt"))

criterion = nn.CrossEntropyLoss()

# Run on test data.
test_data = batchify(tensor_generator(corpora["playground_data/john_asv.txt"],word2idx), eval_batch_size)
test_loss = evaluate(test_data, eval_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

num_samples = 25000   # Number of words to be generated 
temperature = 0.8     # Adjust temperature from 0.0 to 1.0 for diversity

chapter = 1
verses = 1
model.eval()         

# No warmup 
hidden = model.init_hidden(1)

# initial_input = torch.LongTensor([[word2idx["<SOC>"]]])
initial_input = torch.randint(vocab_size, (1, 1), dtype=torch.long).to(device)
input = initial_input.to(device)

with open('generated_sample.txt', 'w') as outf:
    
    outf.write("\n\n")
    with torch.no_grad():  # no tracking history
        for i in range(num_samples):
            
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]

            word = idx2word[word_idx.item()]
            
            # Maximum verses in limited to 60 by EDA
            if verses >= 60 and word != "<EOC>":
                word_idx = torch.tensor(word2idx["<SOC>"])
                word = idx2word[word_idx.item()]


            if word == "<EOC>":
                word = "\n\n"
                
            elif word == "<SOC>":
                word = "\n\n" + str(chapter) + "\n\n"
                chapter += 1
                verses = 1

            elif word == "<SOV>":
                word = "\n" + str(verses) + " "
                verses += 1
            elif word == "<EOV>":
                word = "\n"
            else:
                word =  word + ' '

            outf.write(word)
            if (i+1) % 1000 == 0:
                print('|Sampled [{}/{}] words and save to {}|'.format(i+1, num_samples, 'sample.txt'))   
            
            input.fill_(word_idx)
            
            if chapter>28:
                break

# files.download('generated_sample.txt')

def generate_chapter(model, context, temperature=0.5):
    """
    Generate a chapter with given context and model
    """
    
    model.eval()
    
    word = "<SOC>"    # Starting token
    
    initial_input = torch.LongTensor([[word2idx[word]]])
    input = initial_input.to(device)
    
    generated_text = "\n1 "
    verses = 2
    
    with torch.no_grad():  # no tracking history
        
        while word != "<EOC>": 
            output, context = model(input, context)
            word_weights = output.squeeze().div(temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]

            word = idx2word[word_idx.item()]
            
            if verses > 59 and word != "<EOC>":
                break
                
            if word == "<SOC>":
                word = "\n\n"
                break

            if word == "<SOV>":
                word = "\n" + str(verses) + " "
                verses += 1
            elif word == "<EOV>":
                word = ""
            else:
                word =  word + ' '

            generated_text += word
            word = idx2word[word_idx.item()]
            input.fill_(word_idx)
    
    generated_text = generated_text.strip()
    generated_text = generated_text.rstrip("<EOC>")
    generated_text = generated_text.rstrip("59")
    
    generated_text = generated_text.strip()
    return generated_text

# Generate the Gospel of LSTM with your context

gospel_of_lstms = "1\n\n"
context = re.split(" <SOC>", corpora["playground_data/john_asv.txt"])
for i, text in enumerate(context):
    if not text.startswith("<SOC>"):
        text = "<SOC> " + text
    text = text.strip()
    text = batchify(tensor_generator(text, word2idx), 1)
    warmup_state = get_warmup_state(text)
#     warmup_state = model.init_hidden(1)
    
    output = generate_chapter(model, warmup_state, temperature= 0.5)
    gospel_of_lstms += output + "\n\n" + str(i+2) + "\n\n"
    
    print("{}th chapter is generated".format(i+1))

gospel_of_lstms = gospel_of_lstms.strip()

with open("gospel_of_lstms.txt", "w") as text_file:
    text_file.write(gospel_of_lstms)

# files.download('gospel_of_lstms.txt')
