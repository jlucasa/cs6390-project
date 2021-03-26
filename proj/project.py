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

problematic_chars = {
    '``': '"',
    "''": '"',
    '"': '"'
}


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


# def lint_sentence_for_vocab(sent):
#     sent = sent.lower()
#     words = word_tokenize(sent)
#     filtered = [lemmatizer.lemmatize(word) for word in words]
#     return list(set(filtered))


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

    # for i, val in enumerate(tokens):
    #     if i == start:
    #         if i == end:
    #             if curr_tags[i] == 'O':
    #                 to_ret.append(f'U-{technique}')
    #             else:
    #                 to_ret.append(curr_tags[i])
    #         else:
    #             to_add = [
    #                 ''
    #             ]
    # i = 0
    # while i < len(tokens):
    #     if i == start:
    #         if i == end:
    #             if curr_tags[i] == 'O':
    #                 to_ret.append(f'U-{technique}')
    #             else:
    #                 to_ret.append(curr_tags[i])
    #         else:
    #             if curr_tags[i] == 'O':
    #                 to_ret.append(f'B-{technique}')
    #             else:
    #                 to_ret.append(curr_tags[i])
    #
    #             j = i + 1
    #             while j <= end:
    #                 if j == end:
    #                     if curr_tags[i] == 'O':
    #                         to_ret.append(f'L-{technique}')
    #                     else:
    #                         to_ret.append(curr_tags[i])
    #                 else:
    #                     if curr_tags[i] == 'O':
    #                         to_ret.append(f'I-{technique}')
    #                     else:
    #                         to_ret.append(curr_tags[i])
    #
    #                 j += 1
    #
    #             i = j
    #     else:
    #         to_ret.append(curr_tags[i])
    #         i += 1
    #
    # return to_ret


def convert_start_and_end(target, tokens, start, end, should_remove_leading_and_trailing_invalid_words=True):
    token_spans = [(tok.idx, tok.idx + len(tok.text)) for tok in tokens]

    try:
        new_start = token_spans.index(next(filter(lambda sp: sp[0] == start, token_spans)))
    except Exception:
        print('wtfffff')
    try:
        new_end = token_spans.index(next(filter(lambda sp: sp[1] == end, token_spans))) + 1
    except Exception:
        print('aghhhhhh')

    if tokens[new_start:new_end].text != target.text:
        raise Exception(
            f'Mismatch between tokenized span and character span: target was "{target}" and ' +
            f'tokenized span was "{tokens[new_start:new_end]}"'
        )

    # sentence_tokens = sent_tokenize(sent)
    #
    # tokenized_sents = list(tokenizer.span_tokenize_sents(sentence_tokens))
    # token_spans = [(st, en) for st, en in tokenized_sents[0]]
    #
    # for i in range(1, len(tokenized_sents)):
    #     offset = token_spans[-1][1] + 1
    #     while sent[offset].isspace():
    #         offset += 1
    #
    #     token_spans.extend([(st + offset, en + offset) for st, en in tokenized_sents[i]])
    #
    # # token_spans = list(tokenizer.span_tokenize(sent))
    # token_spans_as_txt = [sent[st:en] for st, en in token_spans]
    #
    # if token_spans_as_txt != tokens:
    #     differences_spans = np.setdiff1d(token_spans_as_txt, tokens)
    #     differences_orig = np.setdiff1d(tokens, token_spans_as_txt)
    #
    #     if len(list(filter(lambda x: x not in problematic_chars, differences_spans))) > 0:
    #         print('hello world')
    #         # raise Exception('Mismatch in sentence span tokens')
    #
    #     if len(list(filter(lambda x: x not in problematic_chars, differences_orig))) > 0:
    #         raise Exception('Mismatch in original tokens')
    #
    #     # for i in range(len(token_spans_as_txt)):
    #     #     if token_spans_as_txt[i] in problematic_chars:
    #     #         token_spans_as_txt[i] = problematic_chars[token_spans_as_txt[i]]
    #     #
    #     # for i in range(len(tokens)):
    #     #     if tokens[i] in problematic_chars:
    #     #         tokens[i] = problematic_chars[tokens[i]]
    #
    # try:
    #     new_start = token_spans.index(next(filter(lambda sp: sp[0] == start, token_spans)))
    # except Exception:
    #     print('why god')
    # try:
    #     new_end = token_spans.index(next(filter(lambda sp: sp[1] == end, token_spans))) + 1
    # except Exception:
    #     print('this is the definition of hell')
    #
    if should_remove_leading_and_trailing_invalid_words:
        while not new_start >= len(tokens) - 1 and not is_valid_word(tokens[new_start]):
            new_start += 1

        while not new_end <= 0 and not is_valid_word(tokens[new_end - 1]):
            new_end -= 1

    return new_start, new_end


def main():
    training_df = load_df_from_file('./data/training_set_task2.txt.test')
    testing_df = load_df_from_file('./data/test_set_task2.txt')
    dev_df = load_df_from_file('./data/dev_set_task2.txt')
    techniques_from_file = load_from_file('./data/techniques_list_task1-2.txt')
    techniques_sequence = defaultdict(lambda: -1)

    techniques_abbrev_map = {
        x: ''.join(e[0] if e[0].isalpha() else e[1] for e in x.split()).upper()
        for x in techniques_from_file
    }

    techniques_abbrev_map['Slogans'] = 'Sl'
    techniques_abbrev_map['Smears'] = 'Sm'

    training_df['filtered_text'] = [[] for _ in range(len(training_df))]
    training_df['tags'] = [[] for _ in range(len(training_df))]
    training_df['orth_mappings'] = [[] for _ in range(len(training_df))]
    training_df['tag_mappings'] = [[] for _ in range(len(training_df))]

    testing_df['filtered_text'] = [[] for _ in range(len(testing_df))]

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

    vocab_train_sequence = {}

    print('Converting all text entries to SpaCy Doc Objects')

    two_orth = 11711838292424000352

    vocab = defaultdict(lambda: -1)

    for index, row in training_df.iterrows():
        text = training_df.loc[index, 'text'].lower()
        doc_text = nlp_engine(text)
        filtered_text = filter_sentence(doc_text)

        orth_mappings = [model_and_utils.orth_mapping(tok.text, tok.orth) for tok in filtered_text]

        for orth_map in orth_mappings:
            if vocab[orth_map.tok] != -1:
                if orth_map.orth != vocab[orth_map.tok]:
                    raise Exception(
                        f'Observed bad orth value for "{orth_map.tok}": Expected orth was ' +
                        f'{vocab[orth_map.tok]}, found {orth_map.orth}'
                    )
            else:
                vocab[orth_map.tok] = orth_map.orth

        # print([tok.orth for tok in filtered_text])
        #
        # split_text = doc_text.text.split()
        #
        # if 'two' in split_text:
        #     print([tok.orth for tok in filtered_text if tok.text == 'two'])
        #     print(two_orth)

        training_df.loc[index, 'filtered_text'] = filtered_text
        training_df.loc[index, 'tags'] = ['O' if is_valid_word(x) else 'SW' for x in doc_text]
        training_df.loc[index, 'text'] = doc_text
        training_df.loc[index, 'orth_mappings'] = orth_mappings

        label_set = row['labels']
        for i in range(len(label_set)):
            label = label_set[i]
            start = label['start']
            end = label['end']
            label_doc_tf = doc_text.char_span(start, end)

            row['labels'][i].update({'text_fragment': label_doc_tf})

    print('Converting all character start and end indices to token start and end indices')

    for index, row in training_df.iterrows():
        label_set = row['labels']
        text = row['text']
        filtered_text = row['filtered_text']

        for i in range(len(label_set)):
            label = label_set[i]
            start = label['start']
            end = label['end']
            tags = training_df.loc[index, 'tags']
            fragment = label['text_fragment']

            new_start, new_end = convert_start_and_end(fragment, text, start, end)
            row['labels'][i].update({'token_start': new_start, 'token_end': new_end})

            if len(list(filter(lambda tag: tag != 'O' and tag != 'SW', tags[new_start:new_end+1]))) > 0 \
                    or new_start >= new_end:
                continue

            abbreviated_technique = techniques_abbrev_map[label['technique']]

            tags = get_tags(
                label['token_start'],
                label['token_end'],
                abbreviated_technique,
                row['tags'])

            if len(tags) != len(text):
                raise Exception('Invalid generation of tags for text')

            training_df.loc[index, 'tags'] = tags

        training_df.loc[index, 'tags'] = list(filter(lambda x: x != 'SW', training_df.loc[index, 'tags']))
        training_df.loc[index, 'tag_mappings'] = [
            model_and_utils.orth_mapping(tag, techniques_sequence[tag]) for tag in training_df.loc[index, 'tags']
        ]

        if len(training_df.loc[index, 'tags']) != len(training_df.loc[index, 'filtered_text']):
            raise Exception('Invalid generation of filtered tags for filtered text')

        if len(training_df.loc[index, 'tag_mappings']) != len(training_df.loc[index, 'filtered_text']):
            raise Exception('Invalid generation of filtered tag mappings for filtered text')

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
    epochs = model_and_utils.NUM_EPOCHS

    lstm_model.train()

    for epoch in range(epochs):
        print(f'running for epoch {epoch}')
        for index, row in training_df.iterrows():
            lstm_model.zero_grad()

            label_set = row['labels']
            sent = get_lemmas_of_sent(row['text'])
            input_sent = model_and_utils.get_tensor_from_sentence(sent, vocab_train_sequence)

            for label in label_set:
                tags = label['bio_tags']
                input_tags = model_and_utils.get_tensor_from_sentence(tags, techniques_sequence)

                tag_scores = lstm_model(input_sent)

                nll_loss = loss_function(tag_scores, input_tags)
                nll_loss.backward()

                optimizer.step()

    lstm_model.eval()

    for index, row in testing_df.iterrows():
        sent = get_lemmas_of_sent(row['text'])
        input_sent = model_and_utils.get_tensor_from_sentence(sent, vocab_train_sequence)

        print(model_and_utils.make_prediction(lstm_model, input_sent, reverse_techniques_sequence))


if __name__ == '__main__':
    main()
