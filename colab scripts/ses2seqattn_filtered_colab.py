from keras import backend as K
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
import tensorflow as tf
import tarfile
import wget
import re
import copy
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
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


def get_json_data(remove_id=True, sample_size=20000, start_token='\t', end_token='\n'):
    dataset = []
    dataset_name = wget.download(
        "https://s3.us-east-2.amazonaws.com/leclair.tech/data/funcom/funcom_filtered.tar.gz")
    tar = tarfile.open("funcom_filtered.tar.gz", "r:gz")

    for member in tar.getmembers():
        data = []
        size = 0
        if member.name == 'funcom_processed/comments.json' or member.name == 'funcom_processed/functions.json':
            file = tar.extractfile(member)

            data_df = pd.read_json(file.read(), orient='index')
            data_df.reset_index(drop=True, inplace=True)
            data_df.columns = ['Column']

            if member.name == 'funcom_filtered/comments.json':
                data_df['Column'] = start_token + data_df['Column'] + end_token
            size += 1
            dataset.append(data_df['Column'][:sample_size].tolist())

    return dataset


def get_num_words(tokenizer, threshold=10):
    count, total_count = 0, 0

    for key, value in tokenizer.word_counts.items():
        total_count = total_count + 1
        if(value < threshold):
            count = count + 1


data = get_json_data(start_token='', end_token='')
target_data, input_data = data[0], data[1]

df = pd.DataFrame({'Function': [], 'Comments': []})
df['Function'] = input_data
df['Comments'] = target_data

df['Comments'] = '_START_ ' + df['Comments'] + ' _END_'

max_len_code = max([len(txt.split()) for txt in input_data])
max_len_summary = max([len(txt.split()) for txt in target_data]) + 2

x_tr, x_validation, y_tr, y_validation = train_test_split(
    df['Function'], df['Comments'], test_size=0.1, random_state=0, shuffle=True)

# prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

# convert text sequences into integer sequences
x_tr = x_tokenizer.texts_to_sequences(x_tr)
x_validation = x_tokenizer.texts_to_sequences(x_validation)

# padding zero upto maximum length
x_tr = pad_sequences(x_tr,  maxlen=max_len_code, padding='post')
x_validation = pad_sequences(x_validation, maxlen=max_len_code, padding='post')

x_voc_size = len(x_tokenizer.word_index) + 1


# preparing a tokenizer for summary on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))

# convert summary sequences into integer sequences
y_tr = y_tokenizer.texts_to_sequences(y_tr)
y_validation = y_tokenizer.texts_to_sequences(y_validation)

# padding zero upto maximum length
y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
y_validation = pad_sequences(y_validation, maxlen=max_len_summary, padding='post')

y_voc_size = len(y_tokenizer.word_index) + 1


class AttentionLayer(Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape(
                                       (input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape(
                                       (input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape(
                                       (input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        # Be sure to call this at the end
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(
                states, type(states))
            assert isinstance(states, list) or isinstance(
                states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(
                K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>', W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(
                K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>', U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(
                K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh,
                                  self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.zeros_like(inputs)
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            # <= (batch_size, latent_dim
            fake_state = K.tile(fake_state, [1, hidden_size])
            return fake_state

        fake_state_c = create_inital_state(
            encoder_out_seq, encoder_out_seq.shape[-1])
        # <= (batch_size, enc_seq_len, latent_dim
        fake_state_e = create_inital_state(
            encoder_out_seq, encoder_out_seq.shape[1])

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape(
                (input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape(
                (input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]


K.clear_session()
latent_dim = 500

# Encoder
encoder_inputs = Input(shape=(max_len_code,))
encoder_embedding = Embedding(
    x_voc_size, latent_dim, trainable=True)(encoder_inputs)

# LSTM 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output1, state_h1, state_c1 = encoder_lstm1(encoder_embedding)

# LSTM 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# LSTM 3
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# Set up the decoder.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

# LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(
    dec_emb, initial_state=[state_h, state_c])

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

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:],
                    epochs=50, steps_per_epoch=100, callbacks=[es], batch_size=64,
                    validation_data=([x_validation, y_validation[:, :-1]], y_validation.reshape(y_validation.shape[0],
                                                                           y_validation.shape[1], 1)[:, 1:]))

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

# encoder inference
encoder_model = Model(inputs=encoder_inputs, outputs=[
                      encoder_outputs, state_h, state_c])

# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_code, latent_dim))

# Get the embeddings of the decoder sequence
decoder_embedding = dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    decoder_embedding, initial_state=[decoder_state_input_h, decoder_state_input_c])

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


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)
    #print('input_seq: {}, e_out: {} '.format(input_seq,e_out))
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['sostok']

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
        if(sampled_token != 'eostok'):
            decoded_sentence += ' '+sampled_token

            # Exit condition: either hit max length or find stop word.
        if (sampled_token == 'eostok' or len(decoded_sentence.split()) >= (max_len_summary-1)):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        # stop_condition = True
        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def sequence_to_comment(input_seq):
    newString = ''
    for i in input_seq:
        if((i != 0 and i != target_word_index['sostok']) and i != target_word_index['eostok']):
            newString = newString+reverse_target_word_index[i]+' '
    return newString


def sequence_to_code(input_seq):
    newString = ''
    for i in input_seq:
        if(i != 0):
            newString = newString+reverse_source_word_index[i]+' '
    return newString


results = pd.DataFrame(
    {'Function': [], 'Original Comment': [], 'Predicted comment': []})

function, o_comment, p_comment = [], [], []
for i in range(len(x_validation)):
    function.append(sequence_to_code(x_validation[i]))
    o_comment.append(sequence_to_comment(y_validation[i]))
    p_comment.append(decode_sequence(x_validation[i].reshape(1, max_len_code)))

results['Function'] = function
results['Original Comment'] = o_comment
results['Predicted comment'] = p_comment

results.to_csv("results_seq2seq_attention.csv")
