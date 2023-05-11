from tqdm import tqdm
import re
from transformers import BertModel, BertTokenizer
import transformers
from tokenizers import BertWordPieceTokenizer
import numpy as np
import pandas as pd
import os
from pathlib import Path
import time
import logging
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import AdamW, Adam
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

# Hyper parameter
epoch = 20
batch_size = 32
lr = 0.005

loss_list = []
test_acc_list = []
train_acc_list = []

log_path = Path().cwd() / ('LSTM_train_validation_' + 'epoch_'  + '_'+ str(
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
    text = re.sub(r'https?://www\.\S+\.cm', '', text)
    text = re.sub(r'[^a-zA-Z|\s]', '', text)
    text = re.sub(r'\*+', 'swear', text)
    return text

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'',text)

def remove_multiplechars(text):
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    return text

def clean(df):
    for col in ['text']:#,'selected_text']:
        df[col] = df[col].astype(str).apply(lambda x:basic_cleaning(x))
        df[col] = df[col].astype(str).apply(lambda x:remove_html(x))
        df[col] = df[col].astype(str).apply(lambda x:remove_multiplechars(x))
    return df.sample(frac=1)

df_clean = clean(df)
df_clean_selection = df_clean.sample(frac=1)

X = df_clean_selection.text.values
y, uniques = pd.factorize(df_clean_selection.sentiment, sort=True)
y_tf = pd.get_dummies(df_clean_selection.sentiment)
logging.info('-----------Cleaning Done-----------')
print('---------------------------------------------Clean Done-------------------------------------------------------')

fast_tokenizer = BertWordPieceTokenizer('kaggle/working/distilbert_base_uncased/vocab.txt', lowercase=True)
print(fast_tokenizer)
logging.info(f'-----------The tokenizer is {fast_tokenizer}-----------')


def fast_encode(texts, tokenizer, chunk_size=256, maxlen=128):
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(length=maxlen)
    all_ids = []
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    return np.array(all_ids)

texts = df_clean_selection.text.astype(str)
X = fast_encode(
    texts,
    fast_tokenizer,
    maxlen=128
)


class lstm(nn.Module):
    def __init__(self, label_nums):
        super(lstm, self).__init__()
        print("---------------")
        self.embed = nn.Embedding(30524,768)
        self.lstm_layer = nn.LSTM(input_size=768, hidden_size=50, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(input_size=50*2, hidden_size=25, bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(50, 50)
        self.relu = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, label_nums)
        self.reinitialize()

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

    def forward(self, inputs):
        out = self.embed(inputs)
        out,(h,c) = self.lstm_layer(out)
        out,(h,c) = self.lstm_layer2(out)
        out = out.max(axis=1).values
        out=self.drop1(out)
        out=self.fc1(out)
        out=self.relu(out)
        out=self.drop2(out)
        out=self.fc2(out)
        out = F.softmax(out, dim=1)
        return out

#label_num =3
model = lstm(3)

b=torch.cuda.is_available()
if(b):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
optim = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()


def train(train_loader, epoches):
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
            # 正向传播
            optim.zero_grad()
            pred_ =  model(x)
            loss = loss_fn(pred_, y)

            loss_tt += loss
            pred_ = pred_.cpu().detach().numpy()
            right_cnt += np.sum(np.argmax(pred_, axis=1) == y.cpu().detach().numpy() )
            # 反向梯度信息
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # 参数更新
            optim.step()
            iter_num += 1
            if(iter_num % 100 == 0):
                acc_ = accuracy_score(np.argmax(pred_, axis=1), y.cpu().detach().numpy())
                print(f"[ iter_num-{iter_num} ]: loss: {loss:.5f}, acc: {acc_:.3f}, time: {time.time() - start:.1f}")
                logging.info('epoch_no: %d: iter_num- %d, loss %.5f, train_acc %.3f, time %.1f' % (
                    ep, iter_num, loss, acc_, time.time() - start))
                loss_list.append(loss.item())

        loss_tt /= cnt
        acc_ = right_cnt / samples_
        print(f"[ ep: {ep} ] loss_tt: {loss_tt:.5f}, acc: {acc_:.3f}")
        train_acc_list.append(acc_)
        validation(val_dataloader=val_dataloader)

def validation(val_dataloader):
    model.eval()
    cnt = 0
    loss_tt = 0
    right_cnt = 0
    samples_ = 0
    start = time.time()
    for x, y in tqdm(val_dataloader):
        x = x.to(device)
        y = y.to(device)
        cnt += 1
        samples_ += len(y)
        with torch.no_grad():
            pred_ =  model(x)
            loss = loss_fn(pred_, y)

        pred_ = pred_.cpu().detach().numpy()
        loss_tt += loss
        right_cnt += np.sum(np.argmax(pred_, axis=1) == y.cpu().detach().numpy() )

    loss_tt /= cnt
    acc_ = right_cnt / samples_
    print("-------------------------------")
    print(f"loss_tt: {loss_tt:.5f}, valid_acc: {acc_:.3f}, time: {time.time() - start :.1f}")
    print("-------------------------------")
    logging.info('loss_total: %.5f, validation_acc: %.3f, time %.1f' % (loss_tt, acc_, time.time() - start))
    test_acc_list.append(acc_)

# class myDataset(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#
#     # 读取单个样本
#     def __getitem__(self, idx):
#         x = torch.tensor(self.encodings[idx])
#         y = torch.tensor(int(self.labels[idx]))
#         return x, y
#
#     def __len__(self):
#         return len(self.labels)

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
    plt.savefig(('LSTM_' + 'epoch_'+ str(epoch) + '_' + str(
        time.strftime("%m_%d_%H_%M_%S", time.localtime())) + ".jpg"))
    plt.clf()  # Clear figure
    plt.cla()  # Clear axes
    plt.close()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# tr_data = myDataset(X[:-1000,:], y[:-1000])
# val_data = myDataset(X[-1000:,:], y[-1000:])

tr_data = TensorDataset(
    torch.tensor(X_train).long(),
    torch.tensor(y_train).long()
)

val_data = TensorDataset(
    torch.tensor(X_val).long(),
    torch.tensor(y_val).long()
)

# tr_data = TensorDataset(
#     torch.tensor(X[:-1000,:]).long(),
#     torch.tensor(y[:-1000]).long()
# )
# # tr_data = TensorDataset(
# #     torch.tensor(X[:2,:]).long(),
# #     torch.tensor(y[:2]).long()
# # )
# val_data = TensorDataset(
#     torch.tensor(X[-1000:,:]).long(),
#     torch.tensor(y[-1000:]).long()
# )

train_loader = DataLoader(tr_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

train(train_loader=train_loader, epoches=epoch)
validation(val_dataloader=val_dataloader)

plot_save(loss_list, test_acc_list, train_acc_list)
