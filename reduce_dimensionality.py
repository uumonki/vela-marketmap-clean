import numpy as np
from umap import UMAP
import sys


def reduce_dimensionality(filename):

    print('Loading embeddings...')
    embeddings = np.load(f'data/{filename}_embeddings.npy')

    # pca_embeddings = rescale(PCA(n_components=5).fit_transform(embeddings))

    umap_model = UMAP(n_neighbors=15,
                    n_components=5,
                    min_dist=0,
                    metric='cosine',
                    verbose=True,
                    # init=pca_embeddings,
                    random_state=42)

    print('Running UMAP...')
    umap_embeddings = umap_model.fit_transform(embeddings)
    print('Saving...')
    np.save(f'data/{filename}_umap_embeddings.npy', umap_embeddings)
    print(f'Embeddings saved to data/{filename}_umap_embeddings.npy')


if __name__ == '__main__':
    reduce_dimensionality(sys.argv[1])