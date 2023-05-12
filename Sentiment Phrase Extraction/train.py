import os
import torch
import pandas as pd
from scipy import stats
import numpy as np
import time
from tqdm import tqdm
from collections import OrderedDict, namedtuple
import torch.nn as nn
from torch.optim import lr_scheduler
import joblib
from pathlib import Path
import logging
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule
import sys
from sklearn import metrics, model_selection
import torch.nn.functional as F
from torch.utils.data import DataLoader
import re
import warnings
from sklearn.model_selection import train_test_split

# import other python files

import preprocessing
from model import BERTBaseUncased, BERTDatasetTraining
from metrics import jaccard, loss_fn

warnings.filterwarnings("ignore")

'''------------Set Configuration------------'''
BATCH_SIZE = 32
EPOCHS = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SEQUENCE_LENGTH = 108

#Logging
log_path = Path().cwd() / ('BERT_train_valid_test_' + 'epoch_' + str(EPOCHS) + '_' + str(
    time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")

logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log_path,
                    filemode='a')

# Model path/basis
BERT_PATH = 'bert-base-uncased'

# Loading Data
TRAIN_DIR = Path().cwd() / 'Datasets/train.csv'
train = pd.read_csv(TRAIN_DIR)
train = preprocessing.create_targets(train)
logging.info('----------------------------targets created----------------------------')



def run(train):
    '''
    Does train_validation dataset split
    Set learning rate scheduler and optimizer
    Does train_dataset preprocessing, load training dataset, train model
    Does valid_dataset preprocessing, load validation dataset, does model validation
    Train model
    Model validation
    '''
    tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH)

    train_df, valid_df = train_test_split(train, test_size=0.20, random_state=42, shuffle=True)  ## Split Labels

    inputs_train = preprocessing.compute_input_arrays(train_df, 'text', tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH)
    outputs_train = preprocessing.compute_output_arrays(train_df, 'targets')

    train_dataset = BERTDatasetTraining(
        comment_text=inputs_train,
        targets=outputs_train,
        tokenizer=tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
        train=True
    )

    inputs_valid = preprocessing.compute_input_arrays(valid_df, 'text', tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH)
    outputs_valid = preprocessing.compute_output_arrays(valid_df, 'targets')

    valid_dataset = BERTDatasetTraining(
        comment_text=inputs_valid,
        targets=outputs_valid,
        tokenizer=tokenizer,
        max_length=MAX_SEQUENCE_LENGTH,
        train=True
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    valid_loader2 = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BERTBaseUncased(bert_path=BERT_PATH).to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    lr = 3e-5
    num_train_steps = int(len(train_dataset) / BATCH_SIZE * EPOCHS)

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    for epoch in range(EPOCHS):
        print(f'Epoch==>{epoch}')
        logging.info('EPOCH NO: %d' % (epoch))
        train_loop_fn(train_loader, epoch, model, optimizer, device, scheduler=scheduler)
        o, t = eval_loop_fn(valid_loader, epoch, model, device, batch_size=BATCH_SIZE, valshape=valid_df.shape[0])
        eval_loop_fn2(valid_loader2, epoch, model, device, batch_size=BATCH_SIZE, valshape=valid_df.shape[0])
        torch.save(model.state_dict(), "model.bin")


def train_loop_fn(data_loader, epoch, model, optimizer, device, scheduler=None):
    model.train()
    # show progress
    tk0 = tqdm(data_loader, total=len(data_loader))

    jcc_sum, n = 0.0, 0

    for batch_index, batch_data in enumerate(tk0):
        # extract required data from
        ids, mask, token_type_ids, targets = batch_data

        # put on cuda
        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)


        optimizer.zero_grad()
        # output derived after inputting data into model
        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        # Calculate loss
        loss = loss_fn(outputs, targets)

        # Calculate jaccard score
        jc = jaccard(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())

        if batch_index % 100 == 0:
            jcc_sum += jc
            n += 1
            print(f'Training -> training_iter_num-{batch_index}, loss={loss}, jaccard={jc}')
            logging.info('epoch_no: %d: Training -> training_iter_num- %d, loss %.5f, jaccard_score %.5f' %
                         (epoch, batch_index, loss, jc))



        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

    jcc_avg = jcc_sum / n
    print(f'epoch no. {epoch}, jaccard={jcc_avg}')


def eval_loop_fn(data_loader, epoch, model, device, batch_size, valshape):
    model.eval()

    valid_preds = np.zeros((valshape, MAX_SEQUENCE_LENGTH))
    original = np.zeros((valshape, MAX_SEQUENCE_LENGTH))

    for bi, d in enumerate(data_loader):
        ids, mask, token_type_ids, targets = d

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        valid_preds[bi * batch_size: (bi + 1) * batch_size] = outputs.detach().cpu().numpy()
        original[bi * batch_size: (bi + 1) * batch_size] = targets.detach().cpu().numpy()

    # calculates jaccard score
    jc = jaccard(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
    print(f'Validation(Unshuffled) ->  jaccard={jc}')
    logging.info('Validation(Unshuffled)  -> epoch_no: %d, jaccard: %.5f' % (epoch, jc))
    return valid_preds, original

def eval_loop_fn2(data_loader, epoch, model, device, batch_size, valshape):
    model.eval()

    valid_preds = np.zeros((valshape, MAX_SEQUENCE_LENGTH))
    original = np.zeros((valshape, MAX_SEQUENCE_LENGTH))

    for bi, d in enumerate(data_loader):
        ids, mask, token_type_ids, targets = d

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        # valid_preds[bi * batch_size: (bi + 1) * batch_size] = outputs.detach().cpu().numpy()
        # original[bi * batch_size: (bi + 1) * batch_size] = targets.detach().cpu().numpy()

    # calculates jaccard score
    jc = jaccard(outputs.detach().cpu().numpy(), targets.detach().cpu().numpy())
    print(f'Validation(Shuffled) ->  jaccard={jc}')
    logging.info('Validation(Shuffled) -> epoch_no: %d, jaccard: %.5f' % (epoch, jc))
    # return valid_preds, original


