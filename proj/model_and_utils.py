from torch import nn, functional, optim
import torch
import pandas as pd
import json

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
from tensorflow_addons.layers import CRF
import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dropout, TimeDistributed, Bidirectional, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# hyperparameters


class hyperparameters:
    def __init__(self, embed_dim, hidden_dim, vocab_size, tag_size, learning_rate, epochs, batch_size):
        self.EMBED_DIM = embed_dim
        self.HIDDEN_DIM = hidden_dim
        self.VOCAB_SIZE = vocab_size
        self.TAG_SIZE = tag_size
        self.LEARNING_RATE = learning_rate
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size

    # def set_hyperparam_val(self, hyperparam_name, val):
    #     self.__setattr__(hyperparam_name, val)


seed(1)
tensorflow.random.set_seed(2)


class orth_mapping:
    def __init__(self, tok, orth):
        self.tok = tok
        self.orth = orth


class forward_lstm_linear(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, tag_size):
        super(forward_lstm_linear, self).__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embeddings_for_sentence = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)

        self.fc = nn.Linear(hidden_size, tag_size)

    def forward(self, sent):
        sentence_embeddings = self.embeddings_for_sentence(sent)
        output, h_t = self.lstm(sentence_embeddings.view(len(sent), 1, -1))
        prediction_scores = self.fc(output.view(len(sent), -1))
        tag_scores = functional.F.log_softmax(prediction_scores, dim=1)

        return tag_scores


def create_model(hparams):
    """

    :param hparams:
    :type hparams: hyperparameters
    """
    model = Sequential()
    model.add(Embedding(input_dim=hparams.VOCAB_SIZE, output_dim=hparams.HIDDEN_DIM, input_length=hparams.EMBED_DIM))
    model.add(
        Bidirectional(
            LSTM(
                hparams.EMBED_DIM,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            merge_mode='concat'
        )
    )
    model.add(LSTM(hparams.HIDDEN_DIM, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))
    # model.add(Dense(hparams.TAG_SIZE, activation='softmax'))
    # model.add(CRF(hparams.TAG_SIZE))
    model.add(TimeDistributed(Dense(hparams.TAG_SIZE, activation='softmax')))

    adam = Adam(lr=hparams.LEARNING_RATE, beta_1=0.9, beta_2=0.999)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_created_model(model, out_fname):
    plot_model(model, to_file=f'{out_fname}.png', show_shapes=True)


def save_model(model, fname):
    """

    :param model:
    :type model: Sequential
    :param fname:
    :type fname: str
    """

    model.save(fname)


def load_model(fname):
    """

    :param fname:
    :return:
    :rtype: Sequential
    """
    return keras.models.load_model(fname, compile=True)


def train_model_and_get_losses(hparams, tokens, tags, model):
    """

    :param hparams:
    :type hparams: hyperparameters
    :param tokens:
    :type tags: np.array
    :param tags:
    :type tags: np.array
    :param model:
    :type model: Sequential
    """

    history = model.fit(
        x=tokens,
        y=tags,
        batch_size=hparams.BATCH_SIZE,
        verbose=1,
        epochs=hparams.EPOCHS,
        validation_split=0.2
    )

    # history = model.fit(x=tokens, y=tags, batch_size=hparams.BATCH_SIZE, verbose=1, epochs=hparams.EPOCHS, validation_split=0.2)

    # for i in range(hparams.EPOCHS):
    #     fitted = model.fit(x=tokens, y=tags, batch_size=hparams.BATCH_SIZE, verbose=1, epochs=1, validation_split=0.2)
    #     losses.append(fitted.history['loss'][0])

    return history


def get_tensor_from_sentence(sent, mapper):
    mapped = [mapper[word] for word in sent]
    return torch.tensor(mapped, dtype=torch.long)


def init_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.1)


def attempt_gpu_accel():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def make_prediction(model, inputs, tag_map):
    tag_scores = model(inputs)
    predictions = [torch.max(score, 0)[1].item() for score in tag_scores]

    return [tag_map[pred] for pred in predictions]


def plot_losses(history, output_fname):
    plt.title('Loss')
    plt.plot(history['loss'], label='training')
    plt.plot(history['val_loss'], label='validation')
    plt.legend()
    plt.savefig(f'{output_fname}-loss.png', bbox_inches='tight')

    plt.close()

    plt.title('Accuracy')
    plt.plot(history['accuracy'], label='training')
    plt.plot(history['val_accuracy'], label='validation')
    plt.legend()
    plt.savefig(f'{output_fname}-acc.png', bbox_inches='tight')


    # x = np.arange(0, len(losses))
    # y = np.array(losses)
    #
    # plt.plot(x, y)
    # plt.savefig(output_fname, bbox_inches='tight')


def get_padded_tokens_and_tags(df, tag2orth):
    """

    :param df:
    :type df: pd.DataFrame
    :param tag2orth:
    :type tag2orth: defaultdict
    """

    tokens = df['orth_mappings'].to_list()
    max_token_length = max([len(tok_map) for tok_map in tokens])
    tags = df['tag_mappings'].to_list()
    max_tag_length = max([len(tag_map) for tag_map in tags])

    token_orth = [[tok.orth for tok in tok_map] for tok_map in tokens]
    tag_orth = [[tag.orth for tag in tag_map] for tag_map in tags]

    padded_tokens = pad_sequences(token_orth, maxlen=max_token_length, dtype='uint64', padding='post')
    padded_tags = pad_sequences(tag_orth, maxlen=max_tag_length, padding='post', value=tag2orth['O'])
    padded_tags = [to_categorical(tag_map, num_classes=len(tag2orth)) for tag_map in padded_tags]

    return np.array(padded_tokens), np.array(padded_tags)

    # for index, row in df.iterrows():
    #     tokens = row['orth_mappings']
    #     max_token_length = max([len(orth_map) for orth_map in tokens])
    #     padded_tokens = pad_sequences(tokens, maxlen=max_token_length, dtype='long', padding='post')
    #
    #     tags = row['tag_mappings'].to_list()
    #     max_tag_length = max([len(tag_map) for tag_map in tags])
    #     padded_tags = pad_sequences(tags, maxlen=max_tag_length, padding='post', value=tag2orth['O'])
    #     padded_tags = [to_categorical(tag_map, num_classes=len(tag2orth)) for tag_map in padded_tags]
    #
    #     df.loc[index, 'padded_orth_mappings'] = padded_tokens
    #     df.loc[index, 'padded_tag_mappings'] = padded_tags

    # return df


def strip_ws_from_tfs(fname):
    data = {}

    with open(fname, 'r') as file:
        data = json.load(file)

        for entry in data:
            label_set = entry['labels']

            for label in label_set:
                orig_tf = label['text_fragment']
                new_tf = label['text_fragment'].strip()

                if new_tf != orig_tf:
                    print(f'found incorrect text fragment: "{orig_tf}", replacing with "{new_tf}"')
                    label['text_fragment'] = new_tf
                    label['end'] -= (len(orig_tf) - len(new_tf))

    print(f'saving to {fname}.converted')

    with open(f'{fname}.converted', 'w+') as file:
        json.dump(data, file)

    # df = pd.read_json(fname)

    # for index, row in df.iterrows():
    #     label_set = row['labels']
    #
    #     for label in label_set:
    #         orig_tf = label['text_fragment']
    #         new_tf = label['text_fragment'].strip()
    #
    #         if new_tf != orig_tf:
    #             print(f'found incorrect text fragment: "{orig_tf}", replacing with "{new_tf}"')
    #             label['text_fragment'] = new_tf
    #             label['end'] -= (len(orig_tf) - len(new_tf))
    #
    # print(f'saving to {fname}.test')
    # dictionary = df.to_dict()




