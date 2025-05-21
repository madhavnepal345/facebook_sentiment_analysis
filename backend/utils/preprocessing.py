import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words=set(stopwords.words('english'))


    def clean_text(self,text:str)->str:
        # Convert to lowercase
        text=text.lower()

        #REMOVEING URLs
        text=re.sub(r'http\S+|www\S+|https\S+','',text,flages=re.MULTILINE)


        #removing user@ referances and # from hastags

        text=re.sub(r'\@\w+|\#','',text)

        #remove punctuation
        text=re.sub(r'[^\w\s]','',text)

        #removeing numbers
        text=re.sub(r'\d+','',text)

        #tokenizing
        tokens=word_tokenize(text)

        #lemmatiz and stopwords
        tokens=[self.lemmatizer.lemmatize(word)for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    


