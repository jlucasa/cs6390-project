import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from proj import model_and_utils

# hyperparameters

lemmatizer = WordNetLemmatizer()
detokenizer = TreebankWordDetokenizer()
tokenizer = TreebankWordTokenizer()
stopword_set = set(stopwords.words('english'))
nlp_engine = spacy.load('en_core_web_trf')


def load_from_file(fp):
    with open(fp) as file:
        return [line.strip() for line in file.readlines()]


def load_df_from_file(fp):
    return pd.read_json(fp)


def lint_sentence(sent):
    sent = sent.lower()
    words = word_tokenize(sent)
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return detokenizer.detokenize(lemmatized)


def get_lemmas_of_sent(sent):
    sent = sent.lower()
    words = word_tokenize(sent)
    return [lemmatizer.lemmatize(word) for word in words]


def filter_sentence(doc_text):
    original_doc = doc_text[0].doc
    return nlp_engine(' '.join([word.lemma_ for word in original_doc if is_valid_word(word)]))


def is_valid_word(word):
    return word.text not in stopword_set and (word.text.isalpha() or word.text.isdigit())


def get_tags(start, end, technique, curr_tags):
    to_ret = []
    #
    # for i in range(0, start):
    #     to_ret.append(curr_tags[i])
    #
    # for i in range(start, end):
    #     if curr_tags[i] == 'O':
    #         to_ret.append(technique)
    #     else:
    #         to_ret.append(curr_tags[i])
    #
    # for i in range(end, len(curr_tags)):
    #     to_ret.append(curr_tags[i])
    #
    # return to_ret

    if start == end - 1 and curr_tags[start] == 'O':
        to_ret = [f'U-{technique}' if i == start else curr_tags[i] for i in range(len(curr_tags))]
    else:
        for i in range(0, start):
            to_ret.append(curr_tags[i])

        if curr_tags[start] == 'O':
            to_ret.append(f'B-{technique}')
        else:
            to_ret.append(curr_tags[start])

        for i in range(start + 1, end - 1):
            if curr_tags[i] == 'O':
                to_ret.append(f'I-{technique}')
            else:
                to_ret.append(curr_tags[i])

        if curr_tags[end - 1] == 'O':
            to_ret.append(f'L-{technique}')
        else:
            to_ret.append(curr_tags[end])

        for i in range(end, len(curr_tags)):
            to_ret.append(curr_tags[i])

    return to_ret


def convert_start_and_end(target, tokens, start, end, should_remove_leading_and_trailing_invalid_words=True):
    token_spans = [(tok.idx, tok.idx + len(tok.text)) for tok in tokens]

    new_start = token_spans.index(next(filter(lambda sp: sp[0] == start, token_spans)))
    new_end = token_spans.index(next(filter(lambda sp: sp[1] == end, token_spans))) + 1

    if tokens[new_start:new_end].text != target.text:
        raise Exception(
            f'Mismatch between tokenized span and character span: target was "{target}" and ' +
            f'tokenized span was "{tokens[new_start:new_end]}"'
        )

    if should_remove_leading_and_trailing_invalid_words:
        while not new_start >= len(tokens) - 1 and not is_valid_word(tokens[new_start]):
            new_start += 1

        while not new_end <= 0 and not is_valid_word(tokens[new_end - 1]):
            new_end -= 1

    return new_start, new_end


def process_df_and_vocab(df, vocab, techniques, abbrev_map):
    print('Converting all text entries to SpaCy Doc Objects')

    df['filtered_text'] = [[] for _ in range(len(df))]
    df['tags'] = [[] for _ in range(len(df))]
    df['orth_mappings'] = [[] for _ in range(len(df))]
    df['tag_mappings'] = [[] for _ in range(len(df))]
    df['padded_orth_mappings'] = [[] for _ in range(len(df))]
    df['padded_tag_mappings'] = [[] for _ in range(len(df))]

    for index, row in df.iterrows():
        text = df.loc[index, 'text'].lower()
        doc_text = nlp_engine(text)
        filtered_text = filter_sentence(doc_text)

        # orth_mappings = [model_and_utils.orth_mapping(tok.text, tok.orth) for tok in filtered_text]
        #
        # for orth_map in orth_mappings:
        #     if vocab[orth_map.tok] != -1:
        #         if orth_map.orth != vocab[orth_map.tok]:
        #             raise Exception(
        #                 f'Observed bad orth value for "{orth_map.tok}": Expected orth was ' +
        #                 f'{vocab[orth_map.tok]}, found {orth_map.orth}'
        #             )
        #     else:
        #         vocab[orth_map.tok] = orth_map.orth

        df.loc[index, 'filtered_text'] = filtered_text
        df.loc[index, 'tags'] = ['O' if is_valid_word(x) else 'SW' for x in doc_text]
        df.loc[index, 'text'] = doc_text
        # df.loc[index, 'orth_mappings'] = orth_mappings

        label_set = row['labels']
        for i in range(len(label_set)):
            label = label_set[i]
            start = label['start']
            end = label['end']
            label_doc_tf = doc_text.char_span(start, end)

            row['labels'][i].update({'text_fragment': label_doc_tf})

        for tok in filtered_text:
            if vocab[tok.text] == -1:
                vocab[tok.text] = len(vocab)

        df.loc[index, 'orth_mappings'] = [
            model_and_utils.orth_mapping(tok.text, vocab[tok.text]) for tok in filtered_text
        ]

        if -1 in df.loc[index, 'orth_mappings']:
            raise Exception('Bad generation of orth mappings for text')

    print('Converting all character start and end indices to token start and end indices')

    for index, row in df.iterrows():
        label_set = row['labels']
        text = row['text']
        filtered_text = row['filtered_text']

        for i in range(len(label_set)):
            label = label_set[i]
            start = label['start']
            end = label['end']
            tags = df.loc[index, 'tags']
            fragment = label['text_fragment']

            new_start, new_end = convert_start_and_end(fragment, text, start, end)
            row['labels'][i].update({'token_start': new_start, 'token_end': new_end})

            if len(list(filter(lambda tag: tag != 'O' and tag != 'SW', tags[new_start:new_end+1]))) > 0 \
                    or new_start >= new_end:
                continue

            abbreviated_technique = abbrev_map[label['technique']]

            tags = get_tags(
                label['token_start'],
                label['token_end'],
                abbreviated_technique,
                row['tags'])

            if len(tags) != len(text):
                raise Exception('Invalid generation of tags for text')

            df.loc[index, 'tags'] = tags

        df.loc[index, 'tags'] = list(filter(lambda x: x != 'SW', df.loc[index, 'tags']))
        df.loc[index, 'tag_mappings'] = [
            model_and_utils.orth_mapping(tag, techniques[tag]) for tag in df.loc[index, 'tags']
        ]

        if len(df.loc[index, 'tags']) != len(df.loc[index, 'filtered_text']):
            raise Exception('Invalid generation of filtered tags for filtered text')

        if len(df.loc[index, 'tag_mappings']) != len(df.loc[index, 'filtered_text']):
            raise Exception('Invalid generation of filtered tag mappings for filtered text')

    return df, vocab


def get_dfs_and_vocab(training_path, testing_path, vocab, techniques, abbrev_map):
    training_df = load_df_from_file(training_path)
    testing_df = load_df_from_file(testing_path)
    training_df, vocab = process_df_and_vocab(training_df, vocab, techniques, abbrev_map)
    testing_df, vocab = process_df_and_vocab(testing_df, vocab, techniques, abbrev_map)

    return training_df, testing_df, vocab


def main():
    techniques_from_file = load_from_file('./data/techniques_list_task1-2.txt')
    techniques_sequence = defaultdict(lambda: -1)

    techniques_abbrev_map = {
        x: ''.join(e[0] if e[0].isalpha() else e[1] for e in x.split()).upper()
        for x in techniques_from_file
    }

    techniques_abbrev_map['Slogans'] = 'Sl'
    techniques_abbrev_map['Smears'] = 'Sm'
    vocab = defaultdict(lambda: -1)

    # Utilize a BILOU tagging scheme. All tags are mapped to a specific integer value.
    for tech in list(techniques_abbrev_map.values()):
        techniques_sequence[f'B-{tech}'] = len(techniques_sequence)
        techniques_sequence[f'I-{tech}'] = len(techniques_sequence)
        techniques_sequence[f'L-{tech}'] = len(techniques_sequence)
        techniques_sequence[f'U-{tech}'] = len(techniques_sequence)

    # 'O' == any word outside a set of tags. Since it's not technique-specific, a
    # per-technique approach as seen above is not required.
    techniques_sequence['O'] = len(techniques_sequence)

    reverse_techniques_sequence = {val: key for key, val in techniques_sequence.items()}
    reverse_techniques_abbrev_map = {val: key for key, val in techniques_abbrev_map.items()}

    training_df, testing_df, vocab = get_dfs_and_vocab(
        './data/training_set_task2.txt.converted',
        './data/test_set_task2.txt.converted',
        vocab,
        techniques_sequence,
        techniques_abbrev_map
    )

    # training_df, vocab = process_df_and_vocab(training_df, vocab, techniques_sequence, techniques_abbrev_map)
    # testing_df, vocab = process_df_and_vocab(testing_df, vocab, techniques_sequence, techniques_abbrev_map)

    if sys.argv[1] == '--train':
        # dev_df = load_df_from_file('./data/dev_set_task2.txt.converted')

        # training_df.to_json('./data/training_set_task2.txt.use', orient='records')
        # testing_df.to_json('./data/testing_set_task2.txt.use', orient='records')
        # with open('./data/vocab.txt', 'w+') as file:
        #     json.dump(vocab, file)
        # training_df = load_df_from_file('./data/training_set_task2.txt.use')
        # testing_df = load_df_from_file('./data/test_set_task2.txt.use')
        # with open('./data/vocab.txt', 'r') as file:
        #     vocab = json.load(file)

        token_inputs, tag_inputs = model_and_utils.get_padded_tokens_and_tags(training_df, techniques_sequence)
        # training_df = model_and_utils.pad_tokens_and_tags(training_df, techniques_sequence)

        hparams = model_and_utils.hyperparameters(
            embed_dim=len(token_inputs[0]),
            hidden_dim=64,
            vocab_size=len(vocab) + 1,
            tag_size=len(techniques_sequence),
            learning_rate=0.0005,
            epochs=35,
            batch_size=1000
        )

        print('Creating model')

        model = model_and_utils.create_model(hparams)

        print('Training model on training data...')

        history = model_and_utils.train_model_and_get_losses(hparams, token_inputs, tag_inputs, model)

        # for index, row in training_df.iterrows():
        #     tokens = [mapping.orth for mapping in row['padded_orth_mappings']]
        #     tags = [mapping.orth for mapping in row['padded_tag_mappings']]
        #     losses = model_and_utils.train_model_and_get_losses(hparams, tokens, tags, model)

        model_and_utils.plot_created_model(model, './data/model')
        model_and_utils.plot_losses(history.history, output_fname='./data/losses-for-training-data')
        model_and_utils.save_model(model, 'trained-model')
    elif sys.argv[1] == '--test':
        model = model_and_utils.load_model('trained-model')
        token_inputs, tag_inputs = model_and_utils.get_padded_tokens_and_tags(testing_df, techniques_sequence)

        hparams = model_and_utils.hyperparameters(
            embed_dim=len(token_inputs[0]),
            hidden_dim=64,
            vocab_size=len(vocab) + 1,
            tag_size=len(techniques_sequence),
            learning_rate=0.0005,
            epochs=35,
            batch_size=1000
        )

        predicted = model.predict_classes(x=token_inputs)

        for pred in predicted:
            print(pred)

        results = model.evaluate(x=token_inputs, y=tag_inputs, batch_size=hparams.BATCH_SIZE, verbose=1)
        print('test loss, test acc: ', results)
    else:
        print('Input syntax: project.py <--train_or_--test>')
        exit(1)

    # loss_function = model_and_utils.LOSS_FUNCTION
    # optimizer = model_and_utils.init_optimizer(lstm_model)
    # epochs = model_and_utils.NUM_EPOCHS
    #
    # lstm_model.train()
    #
    # for epoch in range(epochs):
    #     print(f'running for epoch {epoch}')
    #     for index, row in training_df.iterrows():
    #         lstm_model.zero_grad()
    #
    #         label_set = row['labels']
    #         sent = get_lemmas_of_sent(row['text'])
    #         input_sent = model_and_utils.get_tensor_from_sentence(sent, vocab_train_sequence)
    #
    #         for label in label_set:
    #             tags = label['bio_tags']
    #             input_tags = model_and_utils.get_tensor_from_sentence(tags, techniques_sequence)
    #
    #             tag_scores = lstm_model(input_sent)
    #
    #             nll_loss = loss_function(tag_scores, input_tags)
    #             nll_loss.backward()
    #
    #             optimizer.step()
    #
    # lstm_model.eval()
    #
    # for index, row in testing_df.iterrows():
    #     sent = get_lemmas_of_sent(row['text'])
    #     input_sent = model_and_utils.get_tensor_from_sentence(sent, vocab_train_sequence)
    #
    #     print(model_and_utils.make_prediction(lstm_model, input_sent, reverse_techniques_sequence))


if __name__ == '__main__':
    main()
