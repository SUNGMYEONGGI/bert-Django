# !pip install mxnet
# !pip install transformers==3.0.2                                                                             
# !pip install gluonnlp pandas tqdm                                                 
# !pip install sentencepiece                                                  
# !pip install torch
# !pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
# !pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117



import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import os

#kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

#GPU 사용
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#BERT 모델, Vocabulary 불러오기
bertmodel, vocab = get_pytorch_kobert_model()


import pandas as pd
df = pd.read_excel('./한국어 감정대화데이터셋.xlsx')


df.loc[(df['Emotion'] == "공포"), 'Emotion'] = 0  #공포 => 0
df.loc[(df['Emotion'] == "놀람"), 'Emotion'] = 1  #놀람 => 1
df.loc[(df['Emotion'] == "분노"), 'Emotion'] = 2  #분노 => 2
df.loc[(df['Emotion'] == "슬픔"), 'Emotion'] = 3  #슬픔 => 3
df.loc[(df['Emotion'] == "행복"), 'Emotion'] = 4  #행복 => 4

data_list = []
for q, label in zip(df['Sentence'], df['Emotion'])  :
    data = []
    data.append(q)
    data.append(str(label))

    data_list.append(data)


#train & test 데이터로 나누기
from sklearn.model_selection import train_test_split                                         
dataset_train, dataset_test = train_test_split(data_list, test_size=0.25, random_state=0)


class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


# Setting parameters
max_len = 20 # 100자 미만으로 입력 / 초과시 그 이상은 읽어 들이지 않음
batch_size = 64
warmup_ratio = 0.1
num_epochs = 50
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=7,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
    
    
# model.pt -> 감정분석 bert모델
bert_model = torch.load('./model.pt')


#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    bert_model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = bert_model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("공포가")
            elif np.argmax(logits) == 1:
                test_eval.append("놀람이")
            elif np.argmax(logits) == 2:
                test_eval.append("분노가")
            elif np.argmax(logits) == 3:
                test_eval.append("슬픔이")
            elif np.argmax(logits) == 4:
                test_eval.append("행복이")

        print(test_eval[0] + " 느껴집니다.")

while 1:
    sentence = input("오늘 기분은 어땠니??? : ")
    if sentence == '알아서 뭐하게':
        break
    predict(sentence)
    print("\n")