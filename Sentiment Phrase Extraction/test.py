import pandas as pd
import numpy as np
import transformers
from pathlib import Path
import preprocessing
from model import BERTBaseUncased, BERTDatasetTraining

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

TEST_DIR = Path().cwd() / 'tweet-sentiment-extraction/test.csv'
SUB_DIR = Path().cwd() / 'tweet-sentiment-extraction/sample_submission.csv'

test = pd.read_csv(TEST_DIR)
submission = pd.read_csv(SUB_DIR)

BERT_PATH = 'bert-base-uncased'
tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH)
MAX_SEQUENCE_LENGTH = 108
BATCH_SIZE = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

test['t_text'] = test['text'].apply(lambda x: tokenizer.tokenize(str(x)))

# Start Testing
model = BERTBaseUncased(bert_path=BERT_PATH).to(device)
model.load_state_dict(torch.load("model.bin"))
model.eval()

inputs_test = preprocessing.compute_input_arrays(test, 'text', tokenizer, max_sequence_length=MAX_SEQUENCE_LENGTH)
test_dataset = BERTDatasetTraining(comment_text=inputs_test, tokenizer=tokenizer, max_length=MAX_SEQUENCE_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

with torch.no_grad():
    test_preds = np.zeros((len(test_dataset), MAX_SEQUENCE_LENGTH))

    for bi, d in enumerate(test_loader):
        ids, mask, token_type_ids = d

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)

        outputs = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids
        )

        test_preds[bi * BATCH_SIZE: (bi + 1) * BATCH_SIZE] = outputs.detach().cpu().numpy()

pred = np.where(test_preds >= 0.3, 1, 0)
temp_output = []
for idx, p in enumerate(pred):
    indexes = np.where(p >= 0.3)
    current_text = test['t_text'][idx]
    if len(indexes[0]) > 0:
        start = indexes[0][0]
        end = indexes[0][-1]
    else:
        start = 0
        end = len(current_text)

    temp_output.append(' '.join(current_text[start:end + 1]))

test['temp_output'] = temp_output


def correct_op(row):
    placeholder = row['temp_output']
    for original_token in row['text'].split():
        token_str = ' '.join(tokenizer.tokenize(original_token))
        placeholder = placeholder.replace(token_str, original_token, 1)
    return placeholder


test['temp_output2'] = test.apply(correct_op, axis=1)

## for Neutral tweets keep things same
def replacer(row):
    if row['sentiment'] == 'neutral':
        return row['text']
    else:
        return row['temp_output2']


test['temp_output2'] = test.apply(replacer, axis=1)

submission['selected_text'] = test['temp_output2'].values
submission.to_csv('submission.csv', index=None)
print(submission.head())
