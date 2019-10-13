from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, Flatten
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np

files = [
            'BeckTaylor.txt'
        ]

textOutputFile = 'parsed.txt'

# If you want to save the file, put the name of the output file here
modelOutputFile = 'model.h5'
# If you want to load a previously saved model, put the name of the file here
modelInputFile = ''

def parseFiles(fileList):
    textBlocks = []

    for f in fileList:
        lines = []
        with open(f, 'r', encoding='utf-8') as infile:
            c = 0
            for line in infile:
                c += 1
                lines.append(line.strip())
                print (c)
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
    with open(textOutputFile, 'w') as outfile:
        outfile.write(agg)
    return agg

# Combines the lines from each file into a single array
# Each line is still a single member of that array
def glueFiles(textBlocks):
    agg = []
    for block in textBlocks:
        agg.extend(block)
    #print('{0}'.format(agg))
    with open(textOutputFile, 'w') as outfile:
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
        # print('{0}'.format(token_list))
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, label = input_sequences[:, :-1], input_sequences[:,-1]
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len, total_words

def create_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
    model.add(LSTM(150, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=3, verbose=1, callbacks=[earlystop])

    return model

def generate_text(seed_text, next_words, max_sequence_len, model, tokenizer):
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return seed_text
    
def training(data, tokenizer):
    X, Y, max_len, total_words = dataset_preparation(data, tokenizer)
    model = create_model(X,Y, max_len, total_words)
    if modelOutputFile:
        keras.save(modelOutputFile)
    return max_len, model 


def run():
    tokenizer = Tokenizer()
    blocks = parseFiles(files)
    text = glueFiles(blocks)
    # scope the model variable outside of the conditional
    model = ''
    max_len = 0
    if modelInputFile:
        model = keras.load(modelInputFile)
    else:
        max_len, model = training(text, tokenizer)
    text = generate_text("Dear Whitworth ", 3, max_len, model, tokenizer)
    print(text)



run()
