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

# Software used
<img src="![]()" width="360" height="180">

[//]: # (![image]&#40;https://github.com/mujiexu2/ELEC0130-assignment22-23/blob/main/images/arduinoide.png #pic_left =180x360&#41;)