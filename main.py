import pandas as pd
import numpy as np
import sys
import os 
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
nltk.download('wordnet')
nltk.download('stopwords')
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import string 

class TextIdentifier:
   
    def __init__(self):
        print('Reading files...', end='')
        self.BasePath = "./Training/"
        self.vectorizer = TfidfVectorizer()
        self.Running = True
        self.data = {
            'Austen': 
                ['Mansfield_Park.txt', 'Northanger_Abbey.txt', 'Persuasion.txt', 'Pride_and_Prejudice.txt'],
            'Baum': 
                ['The_Wonderful_Wizard_of_Oz.txt', 'The_Emerald_City_of_Oz.txt', 'Ozma_of_Oz.txt', 'Dorothy_and_the_Wizard_in_Oz.txt'],
            'Verne': 
                 ['All_Around_the_Moon.txt', 'From_the_Earth_to_the_Moon.txt', 'Around_the_World.txt', 'Journey_to_the_Centre_of_the_Earth.txt']
            }
        self.Authors = ['Austen', 'Baum', 'Verne']
        self.df = ''
        self.train = ''
        self.models = {
            'mb': MultinomialNB(),
            'kmeans': KMeans(),
            'lr': LogisticRegression(),
            'svc': SVC(),
            'mlp': MLPClassifier()
        }
        self.readFiles()
        
    @staticmethod     
    def CleanupText(text):
        stop = set(stopwords.words('english') + list(string.punctuation) + ['gutenberg', 'ebook'])
        tokens = word_tokenize(text)
        tokens.pop(0)
        if tokens[0].lower() == 'project':
            tokens.pop(0)
        x = " ".join([i.lower() for i in tokens if i.lower() not in stop and len(i) > 2])
        return x 
  
    @staticmethod
    def RemovePunctuation(text):
        punc = set(string.punctuation)
        tokens = word_tokenize(text)
        x = " ".join([i.lower() for i in tokens if i not in punc])
        return x

    def readFiles(self):
        
        self.df = pd.DataFrame(columns=['Author', 'Text'])
        text = ''
        for key, value in self.data.items():
            for file in value:
                text = ''
                with open(self.BasePath + key + "/" + file, 'r', encoding='utf8') as f:
                    text = text + self.CleanupText(f.read())
                new_row = pd.Series({'Author': key, 'Text': text})
                self.df = pd.concat([self.df, new_row.to_frame().T], ignore_index=True)
        self.df['AuthorEnc'] = LabelEncoder().fit_transform(self.df['Author'])
        print('done')
        print("training...", end='')
        self.runModel()
        
    def readInput(self):
        while True:
            inp = input('What text would you like to Identify? (exit to quit): \n')
            if inp == 'exit':
                exit()
            try:
                with open('./' + inp, 'r', encoding='utf8') as f:
                    self.selection = self.CleanupText(f.read())
            except Exception as E:
                print('Invalid file path or file name try again')
                continue
            else:
                break 
            
    def predict(self):
        print("Predicting...", end='')
        pred = self.models['mb'].predict(self.vectorizer.transform([self.selection]))
        print('done')
        print('The author of this text is: ', self.Authors[pred[0]])
                     
    def runModel(self):
        self.train = self.vectorizer.fit_transform(self.df['Text'])
        self.models['mb'].fit(self.train, self.df['AuthorEnc'] )
        
    def start(self):
        print('done')
        self.readInput()
        self.predict()
        

    def displayResults():
        pass

if __name__ == "__main__":
    print('Tim Williams Text Identifier')
    T = TextIdentifier()
    while T.Running:
        T.start()