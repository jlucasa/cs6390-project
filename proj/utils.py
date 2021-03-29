from collections import defaultdict
import pandas as pd
import json

import spacy
from nltk.corpus import stopwords


stopword_set = set(stopwords.words('english'))


class fileset:
    def __init__(self):
        self.DATA_DIR = ''
        self.IN_MODEL_DIR = ''
        self.OUT_MODEL_DIR = ''
        self.DEV_SET = ''
        self.TECH_SET = ''
        self.TEST_SET = ''
        self.TRAIN_SET = ''
        self.VOCAB_SET = ''
        self.TRAIN_MODEL_LOSSES = ''
        self.MODEL_DESC = ''


class orth_mapping:
    def __init__(self, tok, orth):
        self.tok = tok
        self.orth = orth


def serialize_orth_mapping(mapping):
    """

    :param mapping:
    :type mapping: orth_mapping
    :return:
    :rtype: str
    """

    return f'{mapping.tok},{mapping.orth}'


def deserialize_orth_mapping(to_deserialize):
    """

    :param to_deserialize:
    :type to_deserialize: str
    :return:
    :rtype: orth_mapping
    """

    token_and_orth = to_deserialize.split(',')
    token = token_and_orth[0]
    orth = token_and_orth[1]

    return orth_mapping(token, orth)


def load_from_file(fp):
    with open(fp) as file:
        return [line.strip() for line in file.readlines()]


def load_df_from_file(fp):
    return pd.read_json(fp)


def get_techniques(fileset):
    """

    :param fileset:
    :type fileset: fileset
    :return:
    """
    techniques_from_file = load_from_file(f'{fileset.DATA_DIR}/{fileset.TECH_SET}.txt')
    techniques_sequence = defaultdict(lambda: -1)

    techniques_abbrev_map = {
        x: ''.join(e[0] if e[0].isalpha() else e[1] for e in x.split()).upper()
        for x in techniques_from_file
    }

    techniques_abbrev_map['Slogans'] = 'Sl'
    techniques_abbrev_map['Smears'] = 'Sm'

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

    return [techniques_sequence, techniques_abbrev_map, reverse_techniques_sequence, reverse_techniques_abbrev_map]


def load_nlp_engine(from_language='en_core_web_trf'):
    return spacy.load(from_language)


def filter_sentence(doc_text, engine):
    original_doc = doc_text[0].doc
    return engine(' '.join([word.lemma_ for word in original_doc if is_valid_word(word)]))


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


def convert_start_and_end(target, tokens, start, end, should_remove_leading_and_trailing_invalid_words=True):
    token_spans = [(tok.idx, tok.idx + len(tok.text)) for tok in tokens]

    new_start = token_spans.index(next(filter(lambda sp: sp[0] == start, token_spans)))
    new_end = token_spans.index(next(filter(lambda sp: sp[1] == end, token_spans))) + 1

    if tokens[new_start:new_end].text != target:
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


def get_dfs_and_vocab(training_path, testing_path, dev_path, techniques, abbrev_map, engine):
    vocab = defaultdict(lambda: -1)

    training_df = load_df_from_file(training_path)
    testing_df = load_df_from_file(testing_path)
    dev_df = load_df_from_file(dev_path)
    training_df, vocab = process_df_and_vocab(training_df, vocab, techniques, abbrev_map, engine)
    testing_df, vocab = process_df_and_vocab(testing_df, vocab, techniques, abbrev_map, engine)
    dev_df, vocab = process_df_and_vocab(dev_df, vocab, techniques, abbrev_map, engine)

    return training_df, testing_df, dev_df, vocab


def process_df_and_vocab(df, vocab, techniques, abbrev_map, engine):
    """

    :param df:
    :param vocab:
    :param techniques:
    :param abbrev_map:
    :param engine:
    :type engine: spacy.Language
    :return:
    """
    print('Converting all text entries to SpaCy Doc Objects')

    df['filtered_text'] = [[] for _ in range(len(df))]
    df['tags'] = [[] for _ in range(len(df))]
    df['orth_mappings'] = [[] for _ in range(len(df))]
    df['tag_mappings'] = [[] for _ in range(len(df))]

    for index, row in df.iterrows():
        text = df.loc[index, 'text'].lower()
        doc_text = engine(text)
        filtered_text = filter_sentence(doc_text, engine)

        df.loc[index, 'filtered_text'] = filtered_text
        df.loc[index, 'tags'] = ['O' if is_valid_word(x) else 'SW' for x in doc_text]
        df.loc[index, 'text'] = doc_text

        label_set = row['labels']
        for i in range(len(label_set)):
            label = label_set[i]
            start = label['start']
            end = label['end']
            label_doc_tf = doc_text.char_span(start, end).text

            row['labels'][i].update({'text_fragment': label_doc_tf})

        for tok in filtered_text:
            if vocab[tok.text] == -1:
                vocab[tok.text] = len(vocab)

        df.loc[index, 'orth_mappings'] = [
            orth_mapping(tok.text, vocab[tok.text]) for tok in filtered_text
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
            orth_mapping(tag, techniques[tag]) for tag in df.loc[index, 'tags']
        ]

        if len(df.loc[index, 'tags']) != len(df.loc[index, 'filtered_text']):
            raise Exception('Invalid generation of filtered tags for filtered text')

        if len(df.loc[index, 'tag_mappings']) != len(df.loc[index, 'filtered_text']):
            raise Exception('Invalid generation of filtered tag mappings for filtered text')

    return df, vocab


def serialize_df(df):
    """

    :param df:
    :type df: pd.DataFrame
    :return:
    :rtype: pd.DataFrame
    """

    for index, row in df.iterrows():
        df.loc[index, 'text'] = row['text'].to_bytes().decode('ISO-8859-1')
        df.loc[index, 'filtered_text'] = row['filtered_text'].to_bytes().decode('ISO-8859-1')
        df.loc[index, 'orth_mappings'] = [serialize_orth_mapping(mapping) for mapping in row['orth_mappings']]
        df.loc[index, 'tag_mappings'] = [serialize_orth_mapping(mapping) for mapping in row['tag_mappings']]

    return df


def deserialize_df(df, engine):
    """

    :param df:
    :type df: pd.DataFrame
    :param engine:
    :type engine: spacy.Language
    :return:
    :rtype: pd.DataFrame
    """

    for index, row in df.iterrows():
        df.loc[index, 'text'] = engine.from_bytes(row['text'].encode('ISO-8859-1'))
        df.loc[index, 'filtered_text'] = engine.from_bytes(row['filtered_text'].encode('ISO-8859-1'))
        df.loc[index, 'orth_mappings'] = [deserialize_orth_mapping(entry) for entry in row['orth_mappings']]
        df.loc[index, 'tag_mappings'] = [deserialize_orth_mapping(entry) for entry in row['tag_mappings']]

    return df


def load_processed_dfs_and_vocab(fnames, engine, files, extension='proc'):
    """

    :param fnames:
    :type fnames: list
    :param engine:
    :type engine: spacy.Language
    :param files:
    :type files: fileset
    :param extension:
    :type extension: str
    :return:
    """

    dfs_and_fnames = {}
    for fname in fnames:
        df = load_df_from_file(f'{files.DATA_DIR}/{fname}-{extension}.txt')
        dfs_and_fnames.update({fname: deserialize_df(df, engine)})

    vocab = load_json_from_file(files)

    return dfs_and_fnames, vocab


def write_processed_dfs_and_vocab(dfs_and_fnames, vocab, files, extension='proc'):
    """

    :param dfs_and_fnames:
    :type dfs_and_fnames: dict
    :param vocab:
    :type vocab: defaultdict
    :param files:
    :type files: fileset
    :param extension:
    :type extension: str
    :return:
    """

    for fname in dfs_and_fnames.keys():
        dfs_and_fnames[fname] = serialize_df(dfs_and_fnames[fname])

    write_dfs_and_vocab(dfs_and_fnames, vocab, files, extension)


def write_dfs_and_vocab(dfs_and_fnames, vocab, files, extension):
    """

    :param dfs_and_fnames:
    :type dfs_and_fnames: dict
    :param vocab:
    :type vocab: defaultdict(int)
    :param files:
    :type files: fileset
    :param extension:
    :type extension: str
    :return:
    """

    for df_fname in dfs_and_fnames.keys():
        write_df(df_fname, dfs_and_fnames[df_fname], files, extension)

    write_json_to_file(vocab, files)


def write_df(fname, df, files, extension):
    """

    :param fname:
    :type fname: str
    :param df:
    :type df: pd.DataFrame
    :param files:
    :type files: fileset
    :param extension:
    :type extension: str
    :return:
    """

    df.to_json(f'{files.DATA_DIR}/{fname}-{extension}.txt')


def load_input_file(fname):
    input_lines = load_from_file(fname)
    fnames = {line.split('=')[0]: line.split('=')[1] for line in input_lines}

    to_ret = fileset()

    for fname in fnames:
        setattr(to_ret, fname, fnames[fname])

    return to_ret

    # return fileset(
    #     data_dir=fnames['DATA_DIR'],
    #     in_model_dir=fnames['IN_MODEL_DIR'],
    #     out_model_dir=fnames['OUT_MODEL_DIR'],
    #     dev_set=fnames['DEV_SET'],
    #     tech_set=fnames['TECH_SET'],
    #     train_set=fnames['TRAIN_SET'],
    #     test_set=fnames['TEST_SET'],
    #     vocab_set=fnames['VOCAB_SET'],
    #     model_losses=fnames['MODEL_LOSSES'],
    #     model_desc=fnames['MODEL_DESC']
    # )


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


def write_json_to_file(vocab, files):
    """

    :param vocab:
    :type vocab: defaultdict(int)
    :param files:
    :type files: fileset
    :return:
    """
    with open(f'{files.DATA_DIR}/{files.VOCAB_SET}.txt', 'w+') as file:
        json.dump(vocab, file)


def load_json_from_file(files):
    """

    :param files:
    :type files: fileset
    :return:
    """

    with open(f'{files.DATA_DIR}/{files.VOCAB_SET}.txt', 'r') as file:
        return json.load(file)

