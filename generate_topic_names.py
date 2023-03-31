from bertopic import BERTopic
import pandas as pd
import openai
import re
import sys

prompt = 'I will give you three company descriptions that belong in the same industry. Please come up with a name for the category in which all companies belong in, and be specific with the name while staying concise. Say nothing but the name of the category; do not add the word "Category" in front of your response.'
api_key = '#############################'


def askGPT(text):

    openai.api_key = api_key
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=text,
        temperature=0.4,
        max_tokens=100
    )

    return response.choices[0].text


# If you have access to GPT-3, you can use this function to generate topic names
def generate_topic_names(filepath):

    topic_model = BERTopic.load(f'{filepath}')
    df = topic_model.get_topic_info()

    names = []
    for i in range(0, df.shape[0]):
        # get representative documents
        docs = topic_model.get_representative_docs(i-1)
        print('Calling API for topic ' + str(i-1) + '...')
        name = askGPT(prompt + '\n\nCompany 1: ' + docs[0] + '\n\nCompany 2: ' + docs[1] + '\n\nCompany 3: ' + docs[2])
        # filter punctuation using regex
        name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
        names.append(name)

    df['Label'] = names
    df.to_csv('data/topic_names.csv', index=False)


# Otherwise, use this to generate a spreadsheet of representative documents
def list_representative_docs(filepath):

    topic_model = BERTopic.load(f'{filepath}')
    df = topic_model.get_topic_info()

    docs = []
    for i in range(0, df.shape[0]):
        # get representative documents
        docs.append(topic_model.get_representative_docs(i-1))
    
    # convert 2d list to df
    df = pd.DataFrame(docs)
    df.to_csv('data/representative_docs.csv', index=False)


# Run this on Google Colab to generate topic names from representative documents
def generate_topic_names_from_docs():

    df = pd.read_csv('data/representative_docs.csv')
    names = []
    for i in range(0, df.shape[0]):
        # get representative documents
        docs = df.iloc[i].tolist()
        print('Calling API for topic ' + str(i-1) + '...')
        name = askGPT(prompt + '\n\nCompany 1: ' + docs[0] + '\n\nCompany 2: ' + docs[1] + '\n\nCompany 3: ' + docs[2])
        # filter punctuation using regex
        name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
        names.append(name)

    # export list to csv
    df = pd.DataFrame(names)
    df.to_csv('data/topic_names.csv', index=False)


if __name__ == '__main__':
    list_representative_docs(sys.argv[1])


