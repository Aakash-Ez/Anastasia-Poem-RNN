#!/usr/bin/python
# -*- coding: utf-8 -*-
import io
import json
import torch.nn as nn
import torch.nn.functional as F
from flask_cors import CORS, cross_origin
import random
import numpy as np
import torch
from flask import Flask, jsonify, request

from textblob import TextBlob
def correct_word(word):
  if word=="\n":
    return "\n"
  elif word=="mot":
    return "not"
  elif word=="mo":
    return "no"
  elif word =="j":
    return "i"
  else:
    return str(TextBlob(word).correct())

# Class Definition

class Anastasia(nn.Module):

    def __init__(self, input_size, vocab_size):
        super(Anastasia, self).__init__()
        self.lstm = nn.LSTM(input_size, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 512, batch_first=True)
        self.linear1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.linear2 = nn.Linear(512, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(0.75)
        self.out = nn.Linear(1024, vocab_size)

    def forward(self, input):
        (output, _) = self.lstm(input)
        output = torch.tanh(output)
        output = self.linear1(output)
        output = F.leaky_relu(output)
        output = self.dropout1(output)
        output = self.linear2(output)
        output = F.relu(output)
        output = self.linear3(output)
        output = torch.tanh(output)
        output = self.dropout2(output)
        output = self.out(output)
        output = torch.reshape(output, (input.shape[0], vocab_len))
        return output


char_to_int = {
    '\n': 0,
    ' ': 1,
    '!': 2,
    "'": 3,
    '(': 4,
    ')': 5,
    ',': 6,
    '-': 7,
    '.': 8,
    ':': 9,
    ';': 10,
    '?': 11,
    'a': 12,
    'b': 13,
    'c': 14,
    'd': 15,
    'e': 16,
    'f': 17,
    'g': 18,
    'h': 19,
    'i': 20,
    'j': 21,
    'k': 22,
    'l': 23,
    'm': 24,
    'n': 25,
    'o': 26,
    'p': 27,
    'q': 28,
    'r': 29,
    's': 30,
    't': 31,
    'u': 32,
    'v': 33,
    'w': 34,
    'x': 35,
    'y': 36,
    'z': 37,
    }
int_to_char = {
    0: '\n',
    1: ' ',
    2: '!',
    3: "'",
    4: '(',
    5: ')',
    6: ',',
    7: '-',
    8: '.',
    9: ':',
    10: ';',
    11: '?',
    12: 'a',
    13: 'b',
    14: 'c',
    15: 'd',
    16: 'e',
    17: 'f',
    18: 'g',
    19: 'h',
    20: 'i',
    21: 'j',
    22: 'k',
    23: 'l',
    24: 'm',
    25: 'n',
    26: 'o',
    27: 'p',
    28: 'q',
    29: 'r',
    30: 's',
    31: 't',
    32: 'u',
    33: 'v',
    34: 'w',
    35: 'x',
    36: 'y',
    37: 'z',
    }


# Predict Function Definition

def predict_input(
    model,
    vocab_len,
    newcount=14,
    value='',
    ):
    val = value[:50]
    pattern = [char_to_int[i] for i in val]
    n_val = char_to_int['\n']
    word_out = ''
    count_nl = pattern.count(n_val)
    i = 0
    while count_nl != newcount:
        x = np.reshape(pattern[-50:], (1, 1, 50))
        x = x / float(vocab_len)
        x = torch.tensor(x)
        prediction = F.softmax(model(x))
        index = torch.argmax(prediction).item()
        prediction = np.reshape(prediction.detach().numpy(), vocab_len)
        if int_to_char[index] == '\n' and char_to_int['\n'] \
            in pattern[-40:]:
            while int_to_char[index] == '\n':
                index = np.random.choice(vocab_len, 1, p=prediction)[0]
        if (pattern[-1] == char_to_int[' '] or pattern[-1]
            == char_to_int['\n'] or pattern[-1] == char_to_int['t']) \
            and random.random() < 0.25 or int_to_char[index] == '-':
            index = np.random.choice(vocab_len, 1, p=prediction)[0]
            while int_to_char[index] == '\n' or int_to_char[index] \
                == ' ' or int_to_char[index] == '-':
                index = np.random.choice(vocab_len, 1, p=prediction)[0]
        result = int_to_char[index]
        if result == ' ' and char_to_int['\n'] \
            not in pattern[-40:len(pattern)]:
            result = '\n'
            index = char_to_int['\n']
        if result == '\n':
            count_nl += 1
        seq_in = [int_to_char[value] for value in pattern]
        pattern.append(index)
        word_out += result
        word_out = word_out.replace('\n', ' \n ')
        wordslist = word_out.split()
        if len(wordslist) > 2:
            if wordslist[-1] == wordslist[-2] or len(wordslist[-1]) > 9:
                lastword = wordslist.pop(-1)
                word_out = word_out[:-len(lastword)]
                pattern = pattern[:-len(lastword)].copy()
                i -= len(lastword)
        i += 1
    out = [int_to_char[i].replace('\n', ' \n ') for i in pattern]
    out = ''.join(out)
    out = [correct_word(i) for i in out.split(' ')]
    out = ' '.join(out).replace('\n ', '\n')
    return out


app = Flask(__name__)
cors = CORS(app)
filepath_load = 'ana_weight.pth'
seq_length = 50
vocab_len = len(char_to_int.keys())
model = Anastasia(seq_length, vocab_len).double()
model.load_state_dict(torch.load(filepath_load))


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        content = request.get_json()
        app.logger.info(content)
        inp = str(content.get('sonnet')).lower()
        linecount = int(content.get('numLines'))
        output = predict_input(model, vocab_len, linecount, inp)
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run()
