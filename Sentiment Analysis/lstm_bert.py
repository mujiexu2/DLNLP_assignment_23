# General Libraries
import re
import numpy as np
import pandas as pd
import os
from pathlib import Path
import logging
import time
import sys
import matplotlib.pyplot as plt
# Models & Evaluation
from transformers import BertModel, BertTokenizer
import transformers
from tokenizers import BertWordPieceTokenizer

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset

from torch.optim import Adam
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Data visualization
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit

'''--------------------------- Set Configurations--------------------------------------------------'''
# Hyper parameter
epoch = 20
batch_size = 32
lr = 0.005

loss_list = []
test_acc_list = []
train_acc_list = []

log_path = Path().cwd() / ('LSTM_BERT_train_validation_' + 'epoch_' + str(epoch) + '_'+ str(
    time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".log")

logging.basicConfig(format='%(levelname)s: %(message)s',
                    level=logging.INFO,
                    filename=log_path,
                    filemode='a')

TRAIN_DIR = Path().cwd() / 'tweet-sentiment-extraction/train.csv'
TEST_DIR = Path().cwd() / 'tweet-sentiment-extraction/test.csv'

df = pd.read_csv(TRAIN_DIR)
df_test = pd.read_csv(TEST_DIR)

'''----------------------------Data Cleaning--------------------------------------------------'''


def basic_cleaning(text):
    '''
    Using regular expression to do basic cleaning to text

    Args:
        text: tweet texts

    Returns:
        text which after

    '''
    # Remove any URLs that start with "http://" or "https://" and end with ".cm"
    text = re.sub(r'https?://www\.\S+\.cm', '', text)
    # Remove any characters that are not letters (both upper and lower case) or whitespace characters
    text = re.sub(r'[^a-zA-Z|\s]', '', text)
    text = re.sub(r'\*+', 'swear', text)
    return text


def remove_html(text):
    '''

    Args:
        text:

    Returns:

    '''
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)


# remove repeated characters
def remove_multiplechars(text):
    '''

    Args:
        text:

    Returns:

    '''
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    return text


def clean(df):
    '''

    Args:
        df:

    Returns:

    '''
    for col in ['text']:  # ,'selected_text']:
        df[col] = df[col].astype(str).apply(lambda x: basic_cleaning(x))
        df[col] = df[col].astype(str).apply(lambda x: remove_html(x))
        df[col] = df[col].astype(str).apply(lambda x: remove_multiplechars(x))
    return df.sample(frac=1)


# Clean and standardize data for each column and save it

df_clean = clean(df)
df_clean_selection = df_clean.sample(frac=1)
# df_clean_selection = pd.concat([df_clean.sample(frac=1), df_clean[df_clean.textID.isin(resmaple_id)],
#                                 df_clean[df_clean.textID.isin(resmaple_id)]], axis=0, ignore_index=True)
X = df_clean_selection.text.values
y, uniques = pd.factorize(df_clean_selection.sentiment, sort=True)
y_tf = pd.get_dummies(df_clean_selection.sentiment)
logging.info('-----------Cleaning Done-----------')
print('---------------------------------------------Clean Done-------------------------------------------------------')

# Import bert_tokenizer and save it
bert_tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
save_path = './kaggle/working/distilbert_base_uncased'
if not os.path.exists(save_path):
    os.makedirs(save_path)

bert_tokenizer.save_pretrained(save_path)

fast_tokenizer = BertWordPieceTokenizer('kaggle/working/distilbert_base_uncased/vocab.txt', lowercase=True)
print(fast_tokenizer)
logging.info(f'-----------The tokenizer is {fast_tokenizer}-----------')


#  Fast Encode the cleaned data using the tokenizer mentioned above.
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=128):
    '''

    Args:
        texts:
        tokenizer:
        chunk_size:
        maxlen:

    Returns:

    '''
    # Enable token truncation to a specified maximum length 截断
    tokenizer.enable_truncation(max_length=maxlen)
    # Enable padding of tokens to the same length 填充
    tokenizer.enable_padding(length=maxlen)
    # Initialize an empty list all_ids to store the token ids.
    all_ids = []
    # Iterate over the texts in chunks of size chunk_size, encode each chunk using the encode_batch() method of the
    # tokenizer and store the resulting token ids in the all_ids list.
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i + chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
        #logger.info("Processed chunk %d", i)
    return np.array(all_ids)  # numpy array can do subsequent data processing and manipulation easily


# convert it to string type
texts = df_clean_selection.text.astype(str)

# encoding texts, using tokenizer: fast_tokenizer, setting maximum length: 128 (units of tokens)
X = fast_encode(
    texts,
    fast_tokenizer,
    maxlen=128
)


# Construct model: lstm_bert
class lstm_Bert(nn.Module):
    def __init__(self, label_nums):
        super(lstm_Bert, self).__init__()
        # Initialization starts with transformer_layer (pre-trained model : distilbert-base-uncased)
        transformer_layer = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        p = next(transformer_layer.parameters())
        self.embed = nn.Embedding.from_pretrained(p, freeze=True)
        # Add two layers of lstm
        self.lstm_layer = nn.LSTM(input_size=768, hidden_size=50, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(input_size=50 * 2, hidden_size=25, bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(50, 50)
        self.relu = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, label_nums)
        self.reinitialize()

    # initialize parameters
    def reinitialize(self):
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    # Forward propagation
    def forward(self, inputs):
        out = self.embed(inputs)
        #         print('embed: ', out.shape)
        #         out = out[:,0,:].view(-1, 768)
        out, (h, c) = self.lstm_layer(out)
        #         print('lstm_layer: ', out.shape)
        out, (h, c) = self.lstm_layer2(out)
        #         print('lstm_layer2: ', out.shape)
        #         out = out[:,0,:].view(-1, 50)
        out = out.max(axis=1).values
        #         print('max: ', out.shape)
        #         print("self.lstm_layer2:", out.shape)
        out = self.drop1(out)
        #         print('drop1: ', out.shape)
        out = self.fc1(out)
        #         print('fc1: ', out.shape)
        out = self.relu(out)
        out = self.drop2(out)
        #         print('drop2: ', out.shape)
        out = self.fc2(out)
        #         print('fc2: ', out.shape)
        out = F.softmax(out, dim=1)
        return out


# Build Model
model = lstm_Bert(3)

# # Try an instance
# dd = bert_tokenizer.encode('if I were going',
#                            stride=0,
#                            padding=True,
#                            truncation=True, max_length=128)
#
# dd = torch.Tensor([dd]).long()
# pred_ = model(dd)
# # print(dd)

b = torch.cuda.is_available()
if (b):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define the optimization function of the model as Adam and the loss function as nn.CrossEntropyLoss().
model.to(device)
optim = Adam(model.parameters(), lr=0.005)
loss_fn = nn.CrossEntropyLoss()


# Training Function
def train(train_loader, epoches):
    '''

    Args:
        train_loader:
        epoches:

    Returns:

    '''
    #     optim = AdamW(model.parameters(), lr=0.001)
    start = time.time()

    total_train_loss = 0
    iter_num = 0
    for ep in range(epoches):
        cnt = 0
        loss_tt = 0
        right_cnt = 0
        samples_ = 0
        model.train()
        for x, y in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)
            cnt += 1
            samples_ += len(y)
            # Forward propagation
            optim.zero_grad()
            pred_ = model(x)
            loss = loss_fn(pred_, y)

            loss_tt += loss
            pred_ = pred_.cpu().detach().numpy()
            right_cnt += np.sum(np.argmax(pred_, axis=1) == y.cpu().detach().numpy())
            # Backpropagation gradient information
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters
            optim.step()
            iter_num += 1
            if (iter_num % 100 == 0):
                acc_ = accuracy_score(np.argmax(pred_, axis=1), y.cpu().detach().numpy())
                print(f"[ iter_num-{iter_num} ]: loss: {loss:.5f}, acc: {acc_:.3f}, time: {time.time() - start:.1f}")
                logging.info('epoch_no: %d: iter_num- %d, loss %.5f, train_acc %.3f, time %.1f' % (ep, iter_num, loss, acc_, time.time() - start))
                loss_list.append(loss.item())

        loss_tt /= cnt
        acc_ = right_cnt / samples_
        print(f"[ ep: {ep} ] loss_tt: {loss_tt:.5f}, acc: {acc_:.3f}, time: {time.time() - start:.1f}")
        logging.info('epoch_no: %d, loss_total: %.5f, train_acc: %.3f, time %.1f' % (ep, loss, acc_, time.time() - start))
        train_acc_list.append(acc_)
        validation(val_dataloader=val_dataloader)


# Output the results on the validation set
def validation(val_dataloader):
    '''

    Args:
        val_dataloader:

    Returns:

    '''
    start = time.time()
    model.eval()
    cnt = 0
    loss_tt = 0
    right_cnt = 0
    samples_ = 0
    for x, y in tqdm(val_dataloader):
        x = x.to(device)
        y = y.to(device)
        cnt += 1
        samples_ += len(y)
        with torch.no_grad():
            pred_ = model(x)
            loss = loss_fn(pred_, y)

        pred_ = pred_.cpu().detach().numpy()
        loss_tt += loss
        right_cnt += np.sum(np.argmax(pred_, axis=1) == y.cpu().detach().numpy())

    loss_tt /= cnt
    acc_ = right_cnt / samples_
    print("-------------------------------")
    print(f"loss_tt: {loss_tt:.5f}, valid_acc: {acc_:.3f}, time: {time.time() - start:.1f}")
    print("-------------------------------")
    logging.info('loss_total: %.5f, validation_acc: %.3f, time %.1f' % (loss_tt, acc_, time.time() - start))
    test_acc_list.append(acc_)

# class myDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#
#     # Read each sample
#     def __getitem__(self, idx):
#         x = torch.tensor(self.encodings[idx])
#         y = torch.tensor(int(self.labels[idx]))
#         return x, y
#
#     def __len__(self):
#         return len(self.labels)


# tr_data = myDataset(X[:-1000,:], y[:-1000])
# val_data = myDataset(X[-1000:,:], y[-1000:])
# TensorDataset: create a dataset from a set of PyTorch tensors.


# tr_data = TensorDataset(
#     torch.tensor(X[:-1000, :]).long(),
#     torch.tensor(y[:-1000]).long()
# )
# # tr_data = TensorDataset(
# #     torch.tensor(X[:2,:]).long(),
# #     torch.tensor(y[:2]).long()
# # )
# val_data = TensorDataset(
#     torch.tensor(X[-1000:, :]).long(),
#     torch.tensor(y[-1000:]).long()
# )


def plot_save(loss_list, acc_list, train_acc_list):  # 不重要
    '''
    Three figures, validation accuracy vs. epoch no.; training accuracy vs. epoch no.; training loss vs. Batch count
    Args:
        loss_list: training loss for all batches
        acc_list: validation accuracy for 20 epochs
        train_acc_list: training accuracy for 20 epochs

    Returns: three figures

    '''
    x1 = range(1, len(acc_list) + 1)  # set start value to 1
    x2 = range(len(loss_list))
    x3 = range(1, len(train_acc_list) + 1)
    y1 = acc_list
    y2 = loss_list
    y3 = train_acc_list

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(x1, y1, 'o-')
    axs[0].set_title('Validation Accuracy vs. Epoch No.')
    axs[0].set_xlabel('Epochs No.')
    axs[0].set_ylabel('Validation Accuracy')

    axs[1].plot(x3, y3, 'o-')
    axs[1].set_title('Training Accuracy vs. Epoch No.')
    axs[1].set_xlabel('Epochs No.')
    axs[1].set_ylabel('Training dataset Accuracy')

    axs[2].plot(x2, y2, '.-')
    axs[2].set_title('Training Loss vs. Batch Count')
    axs[2].set_xlabel('Batch Count No.')
    axs[2].set_ylabel('Training Loss')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig(('LSTM_BERT' + 'epoch_'+ str(epoch) + '_' + str(
        time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".jpg"))
    plt.clf()  # Clear figure
    plt.cla()  # Clear axes
    plt.close()

# using train_test_split() to split dataset to validation set be 20% and the training set be 80%
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# TensorDataset: create a dataset from a set of PyTorch tensors.
# Use TensorDataset to import training dataset and validation dataset, after converting datasets into TensorDatasets
# each element is a tuple containing a sequence of inputs and a corresponding label,
# so that they can be processed as a whole.
tr_data = TensorDataset(
    torch.tensor(X_train).long(),
    torch.tensor(y_train).long()
)

val_data = TensorDataset(
    torch.tensor(X_val).long(),
    torch.tensor(y_val).long()
)

# Import training data and validation data 导入train数据和validation数据
train_loader = DataLoader(tr_data, batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size, shuffle=False)

# Start Training and validation 开始训练和验证
train(train_loader=train_loader, epoches=epoch)
validation(val_dataloader=val_dataloader)

plot_save(loss_list, test_acc_list, train_acc_list)