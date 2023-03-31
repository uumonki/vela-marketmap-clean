from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import sys


def create_embeddings(filename, sentence_model='all-MiniLM-L6-v2'):

    df = pd.read_csv(f'data/{filename}_preprocessed.csv')

    sentence_model = SentenceTransformer(sentence_model)
    # future work: try other sentence models or train model on jargon

    print("Creating embeddings...")
    embeddings = sentence_model.encode(df['description'].tolist(), show_progress_bar=True)
    np.save(f'data/{filename}_embeddings.npy', embeddings)
    print(f"Embeddings created and saved to data/{filename}_embeddings.npy")


if __name__ == '__main__':
    create_embeddings(sys.argv[1])