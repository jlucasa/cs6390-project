import sys
import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
from nltk import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, TreebankWordTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from proj import projmodel, utils


def main(arguments):
    fileset = utils.load_input_file(arguments.inputfl)
    action = arguments.action

    techniques = utils.get_techniques(fileset)
    nlp_engine = utils.load_nlp_engine()

    techseq = techniques[0]
    techabbrev = techniques[1]
    rev_techseq = techniques[2]
    rev_techabbrev = techniques[3]

    if action == 'proc':
        training_df, testing_df, dev_df, vocab = utils.get_dfs_and_vocab(
            f'{fileset.DATA_DIR}/{fileset.TRAIN_SET}.txt.converted',
            f'{fileset.DATA_DIR}/{fileset.TEST_SET}.txt.converted',
            f'{fileset.DATA_DIR}/{fileset.DEV_SET}.txt.converted',
            techseq,
            techabbrev,
            nlp_engine
        )

        dfs_and_fnames = {
            fileset.TRAIN_SET: training_df,
            fileset.TEST_SET: testing_df,
            fileset.DEV_SET: dev_df
        }

        utils.write_processed_dfs_and_vocab(dfs_and_fnames, vocab, fileset)
    elif action == 'train':
        fnames = [
            fileset.TRAIN_SET,
            fileset.TEST_SET,
            fileset.DEV_SET
        ]

        dfs_and_fnames, vocab = utils.load_processed_dfs_and_vocab(
            fnames,
            nlp_engine,
            fileset
        )

        training_df = dfs_and_fnames[fileset.TRAIN_SET]
        testing_df = dfs_and_fnames[fileset.TEST_SET]
        dev_df = dfs_and_fnames[fileset.DEV_SET]

        token_inputs, tag_inputs = projmodel.get_padded_tokens_and_tags(training_df, techseq)

        hparams = projmodel.hyperparameters(
            embed_dim=len(token_inputs[0]),
            hidden_dim=64,
            vocab_size=len(vocab) + 1,
            tag_size=len(techseq),
            learning_rate=0.0005,
            epochs=35,
            batch_size=1000
        )

        print('Creating model')

        model = projmodel.create_model(hparams)

        print('Training model on training data...')

        history = projmodel.train_model_and_get_losses(hparams, token_inputs, tag_inputs, model)

        projmodel.plot_created_model(model, f'{fileset.DATA_DIR}/{fileset.MODEL_DESC}')
        projmodel.plot_losses(history.history, output_fname=f'{fileset.DATA_DIR}/{fileset.TRAIN_MODEL_LOSSES}')
        projmodel.save_model(model, fileset.OUT_MODEL_DIR)
    elif action == 'test':
        model = projmodel.load_model(fileset.IN_MODEL_DIR)

        fnames = [
            fileset.TRAIN_SET,
            fileset.TEST_SET,
            fileset.DEV_SET
        ]

        dfs_and_fnames, vocab = utils.load_processed_dfs_and_vocab(
            fnames,
            nlp_engine,
            fileset
        )

        training_df = dfs_and_fnames[fileset.TRAIN_SET]
        testing_df = dfs_and_fnames[fileset.TEST_SET]
        dev_df = dfs_and_fnames[fileset.DEV_SET]

        token_inputs, tag_inputs = projmodel.get_padded_tokens_and_tags(testing_df, techseq)

        hparams = projmodel.hyperparameters(
            embed_dim=len(token_inputs[0]),
            hidden_dim=64,
            vocab_size=len(vocab) + 1,
            tag_size=len(techseq),
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
        print(f'ERROR: bad input syntax.')
        exit(1)


if __name__ == '__main__':
    help_str = 'Runs a bidirectional LSTM on data. This LSTM should be able to predict propaganda in text based on 20 ' + \
               'different labels.\nThis project is based off of SEMEVAL 2021 Task 6, Subtask 2. For more information, ' + \
               'you can check out the corpus for task 6 at https://github.com/di-dimitrov/SEMEVAL-2021-task6-corpus.'

    input_helpstr = 'The directory where all input files are stored. Should have the syntax listed in the example ' + \
                    'input text file.'

    action_helpstr = 'What behavior this program should have. ' + \
                     'If training or testing, make sure to run [proc] first.\n' + \
                     '\t [train]=Train the model on a processed dataset loaded in by file, outputting the model.\n' + \
                     '\t [test]=Test the model on a processed dataset loaded in by file. Run [train] before this.\n' + \
                     '\t [proc]=Process datasets as according to provided input paths.'

    parser = argparse.ArgumentParser(description=help_str)

    parser.add_argument('inputfl', type=str, help=input_helpstr)
    parser.add_argument(
        '-A',
        '--action',
        type=str,
        choices=['train', 'test', 'proc'],
        help=action_helpstr
    )

    args = parser.parse_args()

    main(args)
