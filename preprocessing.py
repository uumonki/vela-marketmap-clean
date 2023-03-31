import pandas as pd
import numpy as np
import nltk
import re
import sys
import langdetect


def preprocess(filename, min_length=15, max_length=140, sample=11):

    print("Reading data...")
    df = pd.read_csv(f'data/{filename}.csv')
    df = df.dropna()
    df['description'] = df['description'].astype('str')
    # Remove duplicates
    df = df.drop_duplicates(subset=['description'])
    

    print('Processing data...')
    # Remove descriptions with less than 15 words
    df = df[(df['description'].apply(lambda x: len(re.findall(r'[a-zA-Z]+', x))) <= max_length) 
            & (df['description'].apply(lambda x: len(re.findall(r'[a-zA-Z]+', x))) >= min_length)]
    df = df[df['description'].str.split().str.len().gt(min_length)]

    # Lemmatize if needed
    '''
    nltk.download('stopwords')

    lemmatizer = nltk.WordNetLemmatizer()
    stopwords = set(nltk.corpus.stopwords.words('english'))

    def lemmatize(text):
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        words = text.split(' ')
        return ' '.join([lemmatizer.lemmatize(word.lower()) for word in words if word.lower() not in stopwords])

    # lemmatization and stopword removal
    df['description'] = df['description'].apply(lemmatize)
    '''

    # Filter out non-English descriptions
    def is_english(text):
        try:
            return langdetect.detect(text) == 'en'
        except:
            print(f'Error: {text}')
            return False
        
    df = df[df['description'].apply(is_english)]

    # optional, filters every 12th row to reduce data size
    df = df.iloc[::sample]

    print(f'Saving data to data/{filename}_preprocessed.csv...')
    df.to_csv(f'data/{filename}_preprocessed.csv', index=False)


if __name__ == '__main__':
    preprocess(sys.argv[1])