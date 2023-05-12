# Libraries
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

import transformers
import torch


'''------------Model------------'''
BERT_PATH = 'bert-base-uncased'
tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH)
MAX_SEQUENCE_LENGTH = 108

def create_targets(df):
    '''

    Args:
        df: dataframe needs to have

    Returns:

    '''
    # Does tokenization to the two column: text, selected_text, and convert them to new columns in the same dataframe
    df
    df['t_text'] = df['text'].apply(lambda x: tokenizer.tokenize(str(x)))
    df['t_selected_text'] = df['selected_text'].apply(lambda x: tokenizer.tokenize(str(x)))

    def func(row):
        # assign the column values of 't_text' and 't_selected_text' to x and y, per row
        x, y = row['t_text'], row['t_selected_text'][:]
        for offset in range(len(x)): # when offset = 0, compare text(whole sentence) with selected sentences
            d = dict(zip(x[offset:], y))
            # d= {key = ' I'd have responded, if I were going': value = ' I'd have responded, if I were going'}
            # a list of list comprehension, check = True if k == v, check = False if k not == v
            check = [k == v for k, v in d.items()] # true no. of check == selected text no.
            if all(check) == True:
                break
        return [0] * offset + [1] * len(y) + [0] * (len(x) - offset - len(y))

    df['targets'] = df.apply(func, axis=1)
    return df


def convert_to_bert_inputs(text, tokenizer, max_sequence_length):
    '''
    Does tokenization to the input text, using BERT tokenizer, giving the token ids, attention masks, token type ids for
    further input to BERT model
    Args:
        text:
        tokenizer: the utilized tokenizer
        max_sequence_length: max text sequence length

    Returns: token ids, attention masks, token type ids of the tokenized text

    '''
    # Does tokenization
    inputs = tokenizer.encode_plus(
        text,
        None,
        add_special_tokens=True,
        max_length=max_sequence_length,
    )
    # get the inputs, token ids, token type ids, attention masks
    ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    mask = inputs["attention_mask"]

    # start padding
    padding_length = max_sequence_length - len(ids)
    # done padding to token ids, attention masks, token type ids
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    # return the list
    return [ids, mask, token_type_ids]


def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    '''
    Convert numpy arrays of input data: token ids, attention masks, token type ids to Pytorch tensors
    Args:
        df:
        columns:
        tokenizer:
        max_sequence_length:

    Returns:

    '''
    input_ids, input_masks, input_segments = [], [], []

    for _, instance in tqdm(df.iterrows(), total=len(df)):
        t = str(instance.text)

        ids, masks, segments = convert_to_bert_inputs(t, tokenizer, max_sequence_length)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [
        torch.from_numpy(np.asarray(input_ids, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_masks, dtype=np.int32)).long(),
        torch.from_numpy(np.asarray(input_segments, dtype=np.int32)).long(),
    ]


def compute_output_arrays(df, col):
    '''
    Extracts the values from the col column of the dataframe and converts it into a numpy array
    Args:
        df: the dataframe requires transformation
        col: the column within the dataframe requires transformation

    Returns: the column of the dataframe in numpy array form

    '''
    return np.asarray(df[col].values.tolist())
