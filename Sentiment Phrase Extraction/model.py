import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

MAX_SEQUENCE_LENGTH = 108

class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768 * 2, MAX_SEQUENCE_LENGTH)

    def forward(
            self,
            ids,
            mask,
            token_type_ids
    ):
        # o1 = last_hidden_state = Sequence of hidden-states at the output of the last layer of the model.
        # o2 = pooler_output = for BERT-family of models, this returns the classification token after processing
        # through a linear layer and a tanh activation function
        o1, o2 = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=False)

        apool = torch.mean(o1, 1)
        mpool, _ = torch.max(o1, 1)
        cat = torch.cat((apool, mpool), 1)

        bo = self.bert_drop(cat)
        p2 = self.out(bo)
        p2 = F.sigmoid(p2)

        return p2


# Load Training dataset
class BERTDatasetTraining:
    # initialize dataset basic information
    def __init__(self, comment_text, tokenizer, max_length, targets=None, train=False):
        self.comment_text = comment_text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.targets = targets
        self.train = train

    # return dataset size
    def __len__(self):
        return len(self.comment_text[0])

    # Send data to the model at each iteration
    def __getitem__(self, idx):
        input_ids = self.comment_text[0][idx]
        input_masks = self.comment_text[1][idx]
        input_segments = self.comment_text[2][idx]

        if self.train:
            labels = self.targets[idx]
            return input_ids, input_masks, input_segments, labels

        return input_ids, input_masks, input_segments
