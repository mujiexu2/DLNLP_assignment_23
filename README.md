# DLNLP_assignment_23- 19027325

# Description of the project
This project based on a Kaggle competition, Tweet Sentiment Extraction, aims to extract support phrases for sentiment 
labels. There are of three sentiment labels, positive, neutral and negative.

link: https://www.kaggle.com/competitions/tweet-sentiment-extraction

The competition is separated into two tasks, Sentiment Analysis and Sentiment Phrase Extraction.

Task A: Sentiment Analysis: 

  - LSTM model 
  - Hybrid model, DistilBERT_LSTM model

Task B: Sentiment Phrase Extraction:

  - BERT_BASELINE uncased model

# Role of each file

  - Datasets : This folder saves the datasets given by the Kaggle competition
    - train.csv : train datasets
    - test.csv : test datasets
    - sample_submission.csv : a sample submission file for the competition


  - Sentiment Analysis:
    - lstm.py : the LSTM model
    - lstm_bert.py: the DistilBERT_LSTM model

  - Sentiment Phrase Extraction:
    - main_bert.py : the main file for this BERT_BASE uncased model
    - preprocessing.py : the data preprocessing process, tokenization
    - model.py: 
      - the dataset loaders for training and validation dataset
      - Model
    - train.py :
      - training loop
      - validation loop
      - function runs training and validation function
    - test.py : run testing and output a submission file

# Instructions to Run:
### Run main.py, it will start in the following order:
1. Sentiment Analysis
    - DistilBERT_LSTM
    - LSTM
2. Sentiment Phrase Extraction
    - BERT_Base
### Change the epoch number
To modify the epoch number of Sentiment Analysis models, just change the epoch number in the related python file
To modify the epoch number of Sentiment Phrase Extraction models, the epoch number of train.py and main_bert.py should be change at the same time




# Others
While running, a folder named 'Kaggle' will appear, related to the tokenizer information in order to match the tokenizers
used in LSTM and DistilBERT_LSTM model.

it stores the:
- special_tokens_map.json
- tokenizer.json
- tokenizer_config.json
- vocab.txt

# Software used
<img src="https://github.com/mujiexu2/DLNLP_assignment_23/blob/main/pycharm.png" width="180" height="180">
