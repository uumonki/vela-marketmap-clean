from sentence_transformers import SentenceTransformer
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP
from hdbscan import HDBSCAN

from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction


def rescale(x, inplace=False):
    if not inplace:
        x = np.array(x, copy=True)
    x /= np.std(x[:, 0]) * 10000
    # https://maartengr.github.io/BERTopic/getting_started/tips_and_tricks/tips_and_tricks.html


def fit_model(filename, reduce_dimensions=True, sentence_model='all-MiniLM-L6-v2', cluster_size=30, samples=25):

    print('Loading embeddings...')
    df = pd.read_csv(f'data/{filename}_preprocessed.csv')

    # pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))

    if reduce_dimensions:
        embeddings = np.load(f'data/{filename}_embeddings.npy')
        umap_model = UMAP(n_neighbors=15,
                          n_components=5,
                          min_dist=0,
                          metric='cosine',
                          verbose=True,
                          # init=pca_embeddings,
                          random_state=100)
        
    else:
        embeddings = np.load(f'data/{filename}_umap_embeddings.npy')
        umap_model = BaseDimensionalityReduction()
                    
    hdbscan_model = HDBSCAN(min_cluster_size=cluster_size,    
                            min_samples=samples,
                            prediction_data=True,
                            cluster_selection_method='eom',
                            alpha=1.0)

    sbert_model = SentenceTransformer(sentence_model)

    topic_model = BERTopic(umap_model=umap_model,
                           hdbscan_model=hdbscan_model,
                           embedding_model=sbert_model,
                           language='english',
                           calculate_probabilities=True, 
                           nr_topics=None,
                           top_n_words=20,
                           verbose=True)
  
    print(f'Fitting model with cluster size {cluster_size}...')
    topics, _ = topic_model.fit_transform(df['description'].tolist(), embeddings)

    print("Saving...")
    topic_model.save(f'models/{filename}_{cluster_size}.bin')
    print(f'Model saved to models/{filename}_{cluster_size}.bin')
    df['topic'] = topics
    df.to_csv(f'data/{filename}_with_topics.csv', index=False)
    print(f'Topics saved to data/{filename}_with_topics.csv')
    
    # return number of topics
    return len(topic_model.get_topics())


if __name__ == '__main__':
    fit_model(sys.argv[1], cluster_size=int(sys.argv[2]), samples=int(0.85 * int(sys.argv[2])))