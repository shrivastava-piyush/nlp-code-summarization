import numpy as np

from extraction.data_funcom import get_data, get_characters
from models.seq2seq import Seq2Seq


def generate_data():
    data = get_data()
    target_data, input_data = data[0], data[1]

    characters = get_characters()
    target_characters, input_characters = characters[0], characters[1]

    encoder_tokens = len(input_characters)
    decoder_tokens = len(target_characters)
    encoder_seq_length = max([len(txt.split()) for txt in input_data])
    decoder_seq_length = max([len(txt.split()) for txt in target_data])

    token_indexes_input, token_indexes_target = get_token_indexes(
        input_characters, target_characters)

    encoder_input_data = np.zeros(
        (len(input_data), encoder_seq_length, encoder_tokens), dtype='float32')
    decoder_input_data = np.zeros(
        (len(target_data), decoder_seq_length, decoder_tokens),   dtype='float32')
    decoder_target_data = np.zeros(
        (len(target_data), decoder_seq_length, decoder_tokens),
        dtype='float32')

    for i, (_input, _target) in enumerate(zip(input_data, target_data)):
        for t, char in enumerate(_input):
            encoder_input_data[i, t, token_indexes_input[char]] = 1.
        for t, char in enumerate(_target):
            decoder_input_data[i, t, token_indexes_target[char]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, token_indexes_target[char]] = 1.

    one_hot_data = [encoder_input_data,
                    decoder_input_data, decoder_target_data]
    tokens = [token_indexes_input, token_indexes_target]
    token_lengths = [encoder_tokens, decoder_tokens]
    sequence_lengths = [encoder_seq_length, decoder_seq_length]
    return one_hot_data, input_data, target_data, tokens, token_lengths, sequence_lengths


def get_token_indexes(input_characters,
                      target_characters):
    token_indexes_input = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    token_indexes_target = dict(
        [(char, i) for i, char in enumerate(target_characters)])
    return token_indexes_input, token_indexes_target


def train():
    one_hot_data, input_data, target_data, tokens, token_lengths, sequence_lengths = generate_data()
    model = Seq2Seq(input_data, target_data,
                    one_hot_data[0], one_hot_data[1], one_hot_data[2],
                    tokens[0], tokens[1],
                    token_lengths[0], token_lengths[1],
                    sequence_lengths[0], sequence_lengths[1])
    model.build_model()
    model.train()
    model.apply_inference()
    model.predict()

train()
