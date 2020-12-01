import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense


class Seq2Seq:
    def __init__(self, input_data, output_data,
                 encoder_input_data, decoder_input_data, decoder_target_data,
                 input_token_index, target_token_index,
                 num_encoder_tokens, num_decoder_tokens,
                 encoder_max_seq, decoder_max_seq):

        self.latent_dim = 256
        self.batch_size = 64
        self.epochs = 100

        self.input_data = input_data
        self.output_data = output_data

        self.encoder_input_data = encoder_input_data
        self.decoder_input_data = decoder_input_data
        self.decoder_target_data = decoder_target_data

        self.input_token_index = input_token_index
        self.target_token_index = target_token_index

        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens

        self.encoder_max_seq = encoder_max_seq
        self.decoder_max_seq = decoder_max_seq

    def build_encoder(self):
        self.encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
        encoder = LSTM(self.latent_dim, return_state=True)
        self.encoder_outputs, state_h, state_c = encoder(self.encoder_inputs)
        self.encoder_states = [state_h, state_c]
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

    def build_decoder(self):
        self.decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        self.decoder_lstm = LSTM(
            self.latent_dim, return_sequences=True, return_state=True)
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_inputs,
                                                       initial_state=self.encoder_states)
        self.decoder_dense = Dense(
            self.num_decoder_tokens, activation='softmax')
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

    def build_model(self):
        self.build_encoder()
        self.build_decoder()
        self.model = Model(
            [self.encoder_inputs, self.decoder_inputs], self.decoder_outputs)

    def train(self):
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy')
        self.model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_split=0.2)
        self.model.save('seq2seq.h5')

    def apply_inference(self):
        self.encoder_model = Model(self.encoder_inputs, self.encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        self.decoder_outputs, state_h, state_c = self.decoder_lstm(
            self.decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)
        self.decoder_model = Model(
            [self.decoder_inputs] + decoder_states_inputs,
            [self.decoder_outputs] + decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

    def decode_sequence(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        #target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > self.decoder_max_seq):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def predict(self):
        for seq_index in range(100):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            input_seq = self.encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = self.decode_sequence(input_seq)
            print('-')
            print('Input sentence:', self.input_data[seq_index])
            print('Decoded sentence:', decoded_sentence)
