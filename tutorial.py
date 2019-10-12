from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np

files = [
            'convocation2019.txt'
        ]

outputFile = 'parsed.txt'

def parseFiles(fileList):
    textBlocks = []

    for f in fileList:
        lines = []
        with open(f, 'r') as infile:
            for line in infile:
                lines.append(line.strip())
        textBlocks.append(lines)

    return textBlocks

# Combines the lines from all the files into a single string
def glueBlocks(textBlocks):
    agg = ''
    for block in textBlocks:
        for line in block:
            agg += ' '
            agg += line
    #print('{0}'.format(agg))
    with open(outputFile, 'w') as outfile:
        outfile.write(agg)
    return agg

# Combines the lines from each file into a single array
# Each line is still a single member of that array
def glueFiles(textBlocks):
    agg = []
    for block in textBlocks:
        agg.extend(block)
    #print('{0}'.format(agg))
    with open(outputFile, 'w') as outfile:
        for line in agg:
            outfile.write(line)
    return agg

def test():
    blocks = parseFiles(files)
    glueBlocks(blocks)
    
def dataset_preparation(data, tokenizer):
    #corpus = data.lower().split("\n") 
    corpus = []
    for element in data:
        corpusElement = element.lower()
        corpus.append(corpusElement)
    tokenizer.fit_on_texts(corpus)
    total_words = len(tokenizer.word_index) + 1
    
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        print('{0}'.format(token_list))
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

def create_model():
    pass

def generate_text():
    pass

def run():
    tokenizer = Tokenizer()
    blocks = parseFiles(files)
    text = glueFiles(blocks)
    dataset_preparation(text, tokenizer)

run()


