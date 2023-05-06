from tqdm.auto import tqdm
import re
from transformers import BertModel, BertTokenizer
import transformers
from tokenizers import BertWordPieceTokenizer
import numpy as np 
import pandas as pd 
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertModel


from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import AdamW, Adam
import torch
from sklearn.metrics import accuracy_score


from sklearn.model_selection import StratifiedShuffleSplit
TRAIN_DIR = Path().cwd() / 'tweet-sentiment-extraction/train.csv'
df = pd.read_csv(TRAIN_DIR)

def basic_cleaning(text):
    text = re.sub(r'https?://www\.\S+\.cm', '', text)
    text = re.sub(r'[^a-zA-Z|\s]', '', text)
    text = re.sub(r'\*+', 'swear', text)
    return text

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'',text)

# remove repeated characters
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
# df_clean_selection = pd.concat([df_clean.sample(frac=1), df_clean[df_clean.textID.isin(resmaple_id)],
#                                 df_clean[df_clean.textID.isin(resmaple_id)]], axis=0, ignore_index=True)
X = df_clean_selection.text.values
y, uniques = pd.factorize(df_clean_selection.sentiment, sort=True)
y_tf = pd.get_dummies(df_clean_selection.sentiment)
print('clean Done')


bert_tokenizer = transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")

save_path = './kaggle/working/distilbert_base_uncased'
if not os.path.exists(save_path):
    os.makedirs(save_path)

bert_tokenizer.save_pretrained(save_path)

fast_tokenizer = BertWordPieceTokenizer('kaggle/working/distilbert_base_uncased/vocab.txt', lowercase=True)
print(fast_tokenizer)

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



class lstm_Bert(nn.Module):
    def __init__(self, label_nums):
        super(lstm_Bert, self).__init__()
        transformer_layer = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        p = next(transformer_layer.parameters())
#         self.embed = nn.Embedding(
#             num_embeddings = p.shape[0], 
#             embedding_dim = p.shape[1]
#         )
#         self.embed.weight = p
#         self.embed.weight.requires_grad = False
        self.embed = nn.Embedding.from_pretrained(
            p, freeze=True
        )
        self.lstm_layer = nn.LSTM(input_size=768, hidden_size=50, bidirectional=True)
        self.lstm_layer2 = nn.LSTM(input_size=50*2, hidden_size=25, bidirectional=True)
        self.drop1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(50, 50)
        self.relu = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, label_nums)

#         self.fc = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(50, 50),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
        
#             nn.Linear(50, label_nums)
#         )
        self._reinitialize()

    def _reinitialize(self):
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
#         print('embed: ', out.shape)
#         out = out[:,0,:].view(-1, 768)
        out,(h,c) = self.lstm_layer(out)
#         print('lstm_layer: ', out.shape)
        out,(h,c) = self.lstm_layer2(out)
#         print('lstm_layer2: ', out.shape)
#         out = out[:,0,:].view(-1, 50)
        out = out.max(axis=1).values
#         print('max: ', out.shape)
#         print("self.lstm_layer2:", out.shape)
        out=self.drop1(out)
#         print('drop1: ', out.shape)
        out=self.fc1(out)
#         print('fc1: ', out.shape)
        out=self.relu(out)
        out=self.drop2(out)
#         print('drop2: ', out.shape)
        out=self.fc2(out)
#         print('fc2: ', out.shape)
        out = F.softmax(out, dim=1)
        return out

model = lstm_Bert(3)

dd = bert_tokenizer.encode('if I were going',
                      stride=0,
                      padding=True, 
                      truncation=True, max_length=128)

dd = torch.Tensor([dd]).long()
pred_ = model(dd)
# print(dd)

b=torch.cuda.is_available()
if(b):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)
optim = Adam(model.parameters(), lr=0.005)
loss_fn = nn.CrossEntropyLoss()

def train(train_loader, epoches):
#     optim = AdamW(model.parameters(), lr=0.001)
    model.train()
    total_train_loss = 0
    iter_num = 0
    for ep in range(epoches):
        cnt = 0
        loss_tt = 0
        right_cnt = 0
        samples_ = 0
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
                print(f"[ iter_num-{iter_num} ]: loss: {loss:.5f}, acc: {acc_:.3f}")

        loss_tt /= cnt
        acc_ = right_cnt / samples_
        print(f"[ ep: {ep} ] loss_tt: {loss_tt:.5f}, acc: {acc_:.3f}")

def validation(val_dataloader):
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
            pred_ =  model(x)
            loss = loss_fn(pred_, y)
        
        pred_ = pred_.cpu().detach().numpy()
        loss_tt += loss
        right_cnt += np.sum(np.argmax(pred_, axis=1) == y.cpu().detach().numpy() )

    loss_tt /= cnt
    acc_ = right_cnt / samples_
    print("-------------------------------")
    print(f"loss_tt: {loss_tt:.5f}, acc: {acc_:.3f}")
    print("-------------------------------")

class myDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    # 读取单个样本
    def __getitem__(self, idx):
        x = torch.tensor(self.encodings[idx])
        y = torch.tensor(int(self.labels[idx]))
        return x, y
    
    def __len__(self):
        return len(self.labels)

# tr_data = myDataset(X[:-1000,:], y[:-1000])
# val_data = myDataset(X[-1000:,:], y[-1000:])
tr_data = TensorDataset(
    torch.tensor(X[:-1000,:]).long(), 
    torch.tensor(y[:-1000]).long()
)
# tr_data = TensorDataset(
#     torch.tensor(X[:2,:]).long(), 
#     torch.tensor(y[:2]).long()
# )
val_data = TensorDataset(
    torch.tensor(X[-1000:,:]).long(), 
    torch.tensor(y[-1000:]).long()
)

train_loader = DataLoader(tr_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True)

train(train_loader=train_loader, epoches=10)
validation(val_dataloader=val_dataloader)