# Train, validation, test all in main.py

'''-----------Libraries----------'''
from pathlib import Path
import pandas as pd
import logging
import time
import numpy as np
from tqdm import tqdm
# Model
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# from other py files
import preprocessing
from model import BERTBaseUncased, BERTDatasetTraining
from metrics import jaccard, loss_fn
from train import run

'''----------Configuration----------'''
BATCH_SIZE = 32
EPOCHS = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SEQUENCE_LENGTH = 108

# Logging
log_path = Path().cwd() / ('BERT_train_valid_' + 'epoch_' + str(EPOCHS) + '_' + str(
    time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")

logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log_path,
                    filemode='a')
# Loading Data
TRAIN_DIR = Path().cwd() / 'tweet-sentiment-extraction/train.csv'
# TEST_DIR = Path().cwd() / 'tweet-sentiment-extraction/test.csv'
# SUB_DIR = Path().cwd() / 'tweet-sentiment-extraction/sample_submission.csv'

train = pd.read_csv(TRAIN_DIR)
# test = pd.read_csv(TEST_DIR)
# submission = pd.read_csv(SUB_DIR)

# Model path/basis
BERT_PATH = 'bert-base-uncased'
tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH)
logging.info('-----------Model:Bert-base-uncased-----------')
logging.info(f'-----------The tokenizer is {tokenizer}-----------')
# Preprocessing
# add 'target' column
train = preprocessing.create_targets(train)
print('MAX_SEQ_LENGTH_TEXT', max(train['t_text'].apply(len)))
print('MAX_TARGET_LENGTH', max(train['targets'].apply(len)))
MAX_TARGET_LEN = MAX_SEQUENCE_LENGTH = 108

# does tokenization to test['text'] and train['targets']
# test['t_text'] = test['text'].apply(lambda x: tokenizer.tokenize(str(x)))
train['targets'] = train['targets'].apply(lambda x: x + [0] * (MAX_TARGET_LEN - len(x)))

# train model and validation
logging.info('-----------Start Train & Validation-----------')
run(train)
