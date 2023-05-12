

import sys
sys.path.append('Sentiment Analysis')

# ======================================================================================================================
# Task A: Sentiment Analysis
print('----------Start LSTM model----------')
# The LSTM model
import lstm_fig
print('----------LSTM model done----------')

# The DistilBERT_LSTM model
print('----------Start DistilBERT_LSTM model----------')
import lstm_bert
print('----------DistilBERT_LSTM model done----------')

# ======================================================================================================================
# Task B: Sentiment Phrase Extraction
sys.path.append('Sentiment Phrase Extraction')
print('----------Start BERT_Base uncased model----------')
import main_bert
print('----------BERT_Base uncased model done----------')

