from tabulate import tabulate
import spacy
import json
import pandas as pd

engine = spacy.load('en_core_web_trf')


def load_df_from_file(fp):
    try:
        return pd.read_json(fp)
    except:
        with open(fp) as file:
            return [line.strip() for line in file.readlines()]


def main():
    training_df = load_df_from_file('./data/training_set_task2.txt')
    dev_df = load_df_from_file('./data/dev_set_task2.txt')
    techniques_df = load_df_from_file('./data/techniques_list_task1-2.txt')

    print(techniques_df)


if __name__ == '__main__':
    main()
