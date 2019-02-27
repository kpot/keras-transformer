# Start by importing all the things we'll need.
import keras
import unicodedata
import re
import numpy as np
import os
from models import transformer_nmt_model

# Download the dataset, a set of translated phrases from Spanish to English.
path_to_zip = keras.utils.get_file(
    'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip', 
    extract=True)
path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# LanguageIndex creates a dictionary and reverse-dictionary mapping words to integers and vice-versa,
# given a set of input phrases.
class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx["<pad>"] = 0
        self.idx2word[0] = "<pad>"
        for i,word in enumerate(self.vocab):
            self.word2idx[word] = i + 1
            self.idx2word[i+1] = word

# Converts Unicode text to ASCII            
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Preprocesses a given sentence by turning it into a list of words and punctuation and appending
# start and end tags onto it.
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = "<start> " + w + " <end>"
    return w

# Returns the maximum sequence length for all sequences in the specified t.
def max_length(t):
    return max(len(i) for i in t)

# Reads a dataset of a pair of translated sentences separated by a tab and delimited by the newline
# character. Processes them with the above methods to get a pair of a list of word sequences.
def create_dataset(path, num_examples):
    lines = open(path, encoding="UTF-8").read().strip().split("\n")
    word_pairs = [[preprocess_sentence(w) for w in l.split("\t")] for l in lines[:num_examples]]
    return word_pairs

# Loads the given dataset and processes it into two lists of sequences of word indexes. Also returns
# the dictionaries used to generate these sequences.
def load_dataset(path, num_examples):
    pairs = create_dataset(path, num_examples)
    out_lang = LanguageIndex(sp for en, sp in pairs)
    in_lang = LanguageIndex(en for en, sp in pairs)
    input_data = [[in_lang.word2idx[s] for s in en.split(' ')] for en, sp in pairs]
    output_data = [[out_lang.word2idx[s] for s in sp.split(' ')] for en, sp in pairs]

    max_seq_length = max(max_length(input_data), max_length(output_data))
    input_data = keras.preprocessing.sequence.pad_sequences(input_data, maxlen=max_seq_length, padding="post")
    output_data = keras.preprocessing.sequence.pad_sequences(output_data, maxlen=max_seq_length, padding="post")
    return input_data, output_data, in_lang, out_lang, max_seq_length

# Actually load the data.
num_examples = 118000 # Full example set.
#num_examples = 30000 # Partial set for faster training
input_data, teacher_data, input_lang, target_lang, len_sequence = load_dataset(path_to_file, num_examples)

# This NMT model requires a teacher and a target dataset. The target is always one word ahead of the teacher, forcing
# the model to predict the next word in the sequence at all times. Generate the target next.
target_data = [[teacher_data[n][i+1] for i in range(len(teacher_data[n])-1)] for n in range(len(teacher_data))]
target_data = keras.preprocessing.sequence.pad_sequences(target_data, maxlen=len_sequence, padding="post")

# Shuffle all of the data in unison. This training set has the longest (e.g. most complicated) data at the end,
# so a simple Keras validation split will be problematic if not shuffled.
p = np.random.permutation(len(input_data))
input_data = input_data[p]
teacher_data = teacher_data[p]
target_data = target_data[p]

# Define some hyperparameters used for this model.
vocab_in_size = len(input_lang.word2idx)
vocab_out_size = len(target_lang.word2idx)
epochs = 30
batch_size = 512
heads = 8
dropout = .1
embedding_dim = 256
encoder_depth = 2
decoder_depth = 2
model = transformer_nmt_model(vocab_in_size, vocab_out_size, len_sequence,
                              heads, dropout, embedding_dim, encoder_depth, decoder_depth)

# In order to train against sparse categorical accuracy, Keras requires that the target tensor have a final dimension of 1.
tar_data = np.expand_dims(target_data, axis=-1)
# Train the model.
hist = model.fit([input_data, teacher_data], tar_data,
                 batch_size=batch_size,
                 epochs=epochs,
                 validation_split=0.2)

# Next, we'll show how to use the model for inference. First, we'll need some functions.

# Converts the given sentence (just a string) into a vector of word IDs
# using the language specified. This can be used for either the input (English)
# or target (Spanish) languages.
# Output is 1-D: [timesteps/words]
def sentence_to_vector(sentence, lang):
    pre = preprocess_sentence(sentence)
    vec = np.zeros(len_sequence)
    sentence_list = [lang.word2idx[s] for s in pre.split(' ')]
    for i, w in enumerate(sentence_list):
        vec[i] = w
    return vec

# Given an input string, an encoder model (infenc_model) and a decoder model (infmodel),
# return a translated string.
def translate(input_sentence, model):
    enc_input = sentence_to_vector(input_sentence, input_lang)
    # Reshape so we can use the encoder model. New shape=[samples,sequence length]
    enc_input = enc_input.reshape(1, len(enc_input))
    
    start_vec = target_lang.word2idx["<start>"]
    stop_vec = target_lang.word2idx["<end>"]
    
    cur_vec = np.zeros((1,len_sequence))
    cur_vec[0,0] = start_vec
    cur_word = "<start>"
    output_sentence = ""
    # Start doing the feeding. Terminate when the model predicts an "<end>" or we reach the end
    # of the max target language sentence length.
    i = 0
    while cur_word != "<end>" and i < (len_sequence-1):
        i += 1
        if cur_word != "<start>":
            output_sentence = output_sentence + " " + cur_word
        nvec = model.predict(x=[enc_input, cur_vec])
        # The output of the model is a massive softmax vector with one spot for every possible word. Convert
        # it to a word ID using argmax().
        cur_vec[0,i] = np.argmax(nvec[0,i-1])
        cur_word = target_lang.idx2word[cur_vec[0,i]]
    return output_sentence

# Use the above functions to perform translation given the model
print(translate("I love you.", model))
print(translate("The moon is very large tonight.", model))
print(translate("What is your name?", model))
