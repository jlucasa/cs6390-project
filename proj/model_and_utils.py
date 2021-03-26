from torch import nn, functional, optim
import torch
import pandas as pd
import json

# hyperparameters

EMBEDDING_DIM = 40
HIDDEN_DIM = 40
VOCAB_SIZE = -1
TAG_SIZE = -1
NUM_EPOCHS = 5
LOSS_FUNCTION = nn.NLLLoss()
OPTIMIZER = None


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

    print(f'saving to {fname}.test')

    with open(f'{fname}.test', 'w+') as file:
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




