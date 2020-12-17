import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import re
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K

from attention import AttentionLayer

import sys
sys.path.append('../')
from extraction.data_funcom import get_json_data

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

data = get_json_data(sample_size=5000, start_token='', end_token='')
target_data, input_data = data[0], data[1]

df = pd.DataFrame({'Function': [], 'Comments': []})
df['Function'] = input_data
df['Comments'] = target_data

df['Comments'] = '_START_ ' + df['Comments'] + ' _END_'

max_len_text = max([len(txt.split()) for txt in input_data])
max_len_summary = max([len(txt.split()) for txt in target_data]) + 2

x_tr, x_validation, y_tr, y_validation = train_test_split(
    df['Function'], df['Comments'], test_size=0.1, random_state=0, shuffle=True)

x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

x_tr = x_tokenizer.texts_to_sequences(x_tr)
x_validation = x_tokenizer.texts_to_sequences(x_validation)

x_tr = pad_sequences(x_tr,  maxlen=max_len_text, padding='post')
x_validation = pad_sequences(x_validation, maxlen=max_len_text, padding='post')

x_voc_size = len(x_tokenizer.word_index) + 1


y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

y_tr = y_tokenizer.texts_to_sequences(y_tr)
y_validation = y_tokenizer.texts_to_sequences(y_validation)

y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
y_validation = pad_sequences(y_validation, maxlen=max_len_summary, padding='post')

y_voc_size = len(y_tokenizer.word_index) + 1


K.clear_session()
latent_dim = 500

# Encoder
encoder_inputs = Input(shape=(max_len_text,))
enc_emb = Embedding(x_voc_size, latent_dim, trainable=True)(encoder_inputs)

# LSTM 1
encoder_lstm_1 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output1, state_h1, state_c1 = encoder_lstm_1(enc_emb)

# LSTM 2
encoder_lstm_2 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output2, state_h2, state_c2 = encoder_lstm_2(encoder_output1)

# LSTM 3
encoder_lstm_3 = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm_3(encoder_output2)

# Set up the decoder.
decoder_inputs = Input(shape=(None,))
decoder_embedding_layer = Embedding(y_voc_size, latent_dim, trainable=True)
decoder_embedding = decoder_embedding_layer(decoder_inputs)

# LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
    decoder_embedding, initial_state=[state_h, state_c])

# Attention Layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention output and decoder LSTM output
decoder_concat_input = Concatenate(
    axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()
model.save('models/seq2seqattn')

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=50, steps_per_epoch=100,
                    callbacks=[es], batch_size=64, validation_data=([x_validation, y_validation[:, :-1]], y_validation.reshape(y_validation.shape[0], y_validation.shape[1], 1)[:, 1:]))

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

# encoder inference
encoder_model = Model(inputs=encoder_inputs, outputs=[
                      encoder_outputs, state_h, state_c])

# decoder inference
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_text, latent_dim))

# Get the embeddings of the decoder sequence
decoder_embedding2 = decoder_embedding_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    decoder_embedding2, initial_state=[decoder_state_input_h, decoder_state_input_c])

# attention inference
attn_out_inf, attn_states_inf = attn_layer(
    [decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(
    axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,
                        decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_sequence):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_sequence)
    #print('input_sequence: {}, e_out: {} '.format(input_sequence,e_out))
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
            continue
        sampled_token = reverse_target_word_index[sampled_token_index]
        # print("sampled_token:",sampled_token)
        if(sampled_token != 'end'):
            decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        # stop_condition = True
        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def sequence_to_comment(input_sequence):
    result = ''
    for i in input_sequence:
        if((i != 0 and i != target_word_index['start']) and i != target_word_index['end']):
            result = result+reverse_target_word_index[i]+' '
    return result


def sequence_to_code(input_sequence):
    result = ''
    for i in input_sequence:
        if(i != 0):
            result = result+reverse_source_word_index[i]+' '
    return result


results = pd.DataFrame(
    {'Function': [], 'Original Comment': [], 'Predicted comment': []})

function, o_comment, p_comment = [], [], []
for i in range(len(x_validation)):
    function.append(sequence_to_code(x_validation[i]))
    o_comment.append(sequence_to_comment(y_validation[i]))
    p_comment.append(decode_sequence(x_validation[i].reshape(1, max_len_text)))

results['Function'] = function
results['Original Comment'] = o_comment
results['Predicted comment'] = p_comment

results.to_csv("results_seq2seq_attention.csv")
