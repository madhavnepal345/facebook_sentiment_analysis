from transformers import BertTokenizer,BertForSequenceClassification
from transformers import Trainer,TrainingArguments
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_socre,precision_recall_fscore_support


class SentimentDataset(Dataset):
    def __init__(self,texts,labels,tokeinzer,max_length):
        self.texts=texts
        self.label=label
        self.tokenizer=tokenizer
        self.max_length=max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text= str(self.texts[idx])
        label=self.labels[idx]

        encoding=self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'

        )
        return{
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten()
            'label':torch.tensor(label,dtype=torch.long)
            }