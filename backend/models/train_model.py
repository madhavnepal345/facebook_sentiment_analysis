from transformers import BertTokenizer,BertForSequenceClassification
from transformers import Trainer,TrainingArguments
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.metrics import accuracy_socre,precision_recall_fscore_support
from sklearn.model_selection import train_test_split


class SentimentDataset(Dataset):
    def __init__(self,texts,labels,tokeinzer,max_length):
        self.texts=texts
        self.labels=labels
        self.tokeinzer=tokeinzer
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
        return{'input_ids':encoding['input_ids'].flatten(),
                'attention_mask':encoding['attention_mask'].flatten(),
                'labels':torch.tensor(label,dtype=torch.long)}
    
class BERTModelTrainer:
    def __init__ (self):
        self.device=torch.device("cuda" if torch.cuda.is_available()else"cpu")
        self.tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_map={'negative':0,'neutral':1,'positive':2}
        self.reverse_label_map={v:k for k,v in self.label_map.items()}
    
    def load_data(self,filepath:str):
        # loading the data
        df=pd.read_csv(filepath)
        df['cleaned_text']=df['text'].apply(self.preprocess_text)
        df['label']=df['sentiment'].map(self.label_map)
        return df
    
    def preprocess_text(self,text:str)->str:
        return text.lower().strip()

    def compute_metrics(self,pred):
        labels=pred.label_ids
        pred-pred.predicitions.argmax(-1)
        precision,recall,f1,_= precision_recall_fscore_support(labels,pred,average='weighted')
        acc=accuracy_socre(labels,pred)
        return{
            'accuracy':acc,
            'f1':f1,
            'precision':precision,
            'recall':recall
        }
    def train(self,train_df,eval_df):
        train_datasets=SentimentDataset(
            train_df['cleaned_text'].tolist(),
            train_df['label'].tolist(),
            self.tokenizer,
            max_lenght=128
        )

        eval_dataset=SentimentDataset(
            eval_df['cleaned_text'].tolist(),
            eval_df['label'].tolist(),
            self.tokenizer,
            max_length=128
        )

        #loading the models

        model= BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=len(self.label_map))
        model.to(self.device)


        #training arguments
        training_args=TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )


        # Initialize Trainer

        trainer=Trainer(
            model=model,
            args=training_args,
            train_dataset=train_datasets,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        trainer.train()
        return model
    def save_model(self,model,tokenizer,output_dir):
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    def train_and_save(self,data_path,output_dir="bert_sentiment_model"):
        df=self.load_data(data_path)
        train_df,eval_df=train_test_split(df,test_size=0.2,random_state=42)
        model=self.train(train_df,eval_df)
        self.save_model(model,self.tokenizer,output_dir)