from collections import defaultdict

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import model_and_utils

# hyperparameters

lemmatizer = WordNetLemmatizer()
stopword_set = set(stopwords.words('english'))


def load_df_from_file(fp):
    try:
        return pd.read_json(fp)
    except:
        with open(fp) as file:
            return [line.strip() for line in file.readlines()]


def lint_sentence_for_vocab(sent):
    sent = sent.lower()
    words = word_tokenize(sent)
    filtered = [lemmatizer.lemmatize(word) for word in words]
    return list(set(filtered))


def get_lemmas_of_sent(sent):
    sent = sent.lower()
    # print(sent)
    words = word_tokenize(sent)
    # print(words)
    return [lemmatizer.lemmatize(word) for word in words]


def get_bio_tags(tokens, start, end, technique):
    to_ret = []

    i = 0
    while i < len(tokens):
        if i == start:
            to_ret.append(f'B-{technique}')

            j = i + 1
            while j <= end:
                to_ret.append(f'I-{technique}')
                j += 1

            i = j
        else:
            to_ret.append('O')
            i += 1

    return to_ret


def convert_start_and_end(sent, tokens, start, end):
    substring = sent[start:end]
    substring_tokens = word_tokenize(substring)

    new_start = 0
    new_end = len(tokens) - 1

    if tokens.count(substring_tokens[0]) == 1:
        new_start = tokens.index(substring_tokens[0])
    else:
        token_start_indexes = [i for i, x in enumerate(tokens) if x == substring_tokens[0]]
        for index in token_start_indexes:
            if index >= 0 and tokens[index - 1] not in substring_tokens:
                new_start = index
                break

    if tokens.count(substring_tokens[-1] == 1):
        new_end = tokens.index(substring_tokens[-1])
    else:
        token_end_indexes = [i for i, x in enumerate(tokens) if x == substring_tokens[-1]]
        for index in token_end_indexes:
            if index < len(tokens) - 1 and tokens[index + 1] not in substring_tokens:
                new_end = index
                break

    return new_start, new_end


def main():
    training_df = load_df_from_file('./data/training_set_task2.txt')
    testing_df = load_df_from_file('./data/test_set_task2.txt')
    dev_df = load_df_from_file('./data/dev_set_task2.txt')
    techniques_from_file = load_df_from_file('./data/techniques_list_task1-2.txt')
    techniques_sequence = defaultdict(lambda: -1)

    techniques_abbrev_map = {
        x: ''.join(e[0] if e[0].isalpha() else e[1] for e in x.split()).upper()
        for x in techniques_from_file
    }

    techniques_abbrev_map['Slogans'] = 'Sl'
    techniques_abbrev_map['Smears'] = 'Sm'

    print(techniques_abbrev_map)

    for tech in list(techniques_abbrev_map.values()):
        techniques_sequence[f'B-{tech}'] = len(techniques_sequence)
        techniques_sequence[f'I-{tech}'] = len(techniques_sequence)

    techniques_sequence['O'] = len(techniques_sequence)

    print(techniques_sequence)

    reverse_techniques_sequence = {val: key for key, val in techniques_sequence.items()}
    reverse_techniques_abbrev_map = {val: key for key, val in techniques_abbrev_map.items()}

    # print(techniques_sequence)
    vocab_train_sequence = {}

    # count = 0
    for index, row in training_df.iterrows():
        # count += 1
        # if count == 4:
        #     break

        label_set = row['labels']
        text = row['text']
        tokenized_text = word_tokenize(text.lower())

        for i in range(len(label_set)):
            label = label_set[i]
            start = label['start']
            end = label['end']

            new_start, new_end = convert_start_and_end(text, tokenized_text, start, end)

            # if index == 155:
            #     # print(new_start)
            #     # print(new_end)
            #     # print(start)
            #     # print(end)
            #     # print(tokenized_text)
            row['labels'][i].update({'token_start': new_start, 'token_end': new_end})

            abbreviated_technique = techniques_abbrev_map[label['technique']]
            bio_tags = get_bio_tags(tokenized_text, label['token_start'], label['token_end'], abbreviated_technique)
            # if index == 155:
            #     print(bio_tags)
            label.update({'bio_tags': bio_tags})

        linted_text = lint_sentence_for_vocab(text)
        for word in linted_text:
            if word not in vocab_train_sequence:
                vocab_train_sequence.update({word: len(vocab_train_sequence)})

    for index, row in testing_df.iterrows():
        text = row['text']
        linted_text = lint_sentence_for_vocab(text)

        for word in linted_text:
            if word not in vocab_train_sequence:
                vocab_train_sequence.update({word: len(vocab_train_sequence)})

    model_and_utils.VOCAB_SIZE = len(vocab_train_sequence)
    model_and_utils.TAG_SIZE = len(techniques_sequence)

    lstm_model = model_and_utils.forward_lstm_linear(
        model_and_utils.EMBEDDING_DIM,
        model_and_utils.HIDDEN_DIM,
        model_and_utils.VOCAB_SIZE,
        model_and_utils.TAG_SIZE
        )

    loss_function = model_and_utils.LOSS_FUNCTION
    optimizer = model_and_utils.init_optimizer(lstm_model)
    # proc_device = model_and_utils.attempt_gpu_accel()
    epochs = model_and_utils.NUM_EPOCHS

    # lstm_model.to(proc_device)
    lstm_model.train()

    for epoch in range(epochs):
        print(f'running for epoch {epoch}')

        for index, row in training_df.iterrows():
            # print(f'row {index}')
            lstm_model.zero_grad()

            label_set = row['labels']
            sent = get_lemmas_of_sent(row['text'])
            input_sent = model_and_utils.get_tensor_from_sentence(sent, vocab_train_sequence)

            for label in label_set:
                tags = label['bio_tags']
                # print(len(label['bio_tags']))
                input_tags = model_and_utils.get_tensor_from_sentence(tags, techniques_sequence)

                tag_scores = lstm_model(input_sent)

                nll_loss = loss_function(tag_scores, input_tags)
                nll_loss.backward()

                optimizer.step()

    lstm_model.eval()

    for index, row in testing_df.iterrows():
        sent = get_lemmas_of_sent(row['text'])
        # print(sent)
        input_sent = model_and_utils.get_tensor_from_sentence(sent, vocab_train_sequence)

        print(model_and_utils.make_prediction(lstm_model, input_sent, reverse_techniques_sequence))


if __name__ == '__main__':
    main()
