# Atlas: an automated market mapping tool

Atlas is an NLP-enabled automated market mapping tool. Through text embeddings and clustering, a dataset of 100k company descriptions are classified into approximately 200 categories that roughly partition markets into industries. The trained model can then be used to classify new descriptions into these categories, as well as display a two-dimensional visual that represents the relationships between industries.

## Dependencies

The dependencies of this project are listed in dependencies.txt. Note that CUDA is required to use the pre-trained model.

## Report

It is recommended to read the project report first, in report.pdf.
 
## Using the model

First, download the model from [Google Drive](drive.google.com) and place it in the /models directory. Then, import the Atlas class from atlas.py, which contains most of the commonly used functions to apply the pretrained model. One can also use ```from bertopic import BERTopic``` to use the functions implemented in the BERTopic module (see [documentation](https://maartengr.github.io/BERTopic/index.html)).

The functions are outlined in demo.ipynb. First, instantiate an ```Atlas``` object with parameters ```model_path```, ```data_path```, and ```topic_names_path```.

To classify a description, call ```classify(str)``` to get the name of the category directly, or ```classify_index(str)``` to obtain the index of said category, which can then be used to list all companies in the category using ```get_topic_companies(int)```.

One can also search category names with keywords using ```search_topics(str)```. Finally, ```visualize_topics()``` returns a plotly figure.

## Training the model on your own data

First, make sure that the data has columns 'name' and 'description'. Run preprocessing.py in terminal, passing as argument the file path of the data to preprocess the data. Then, do the same with create_embeddings.py, which should create a file in /data. Optionally, run reduce_dimensionality.py to process the embeddings using UMAP to conduct experiments on clustering, but note that this will disable the model from being able to classify new data (or one can optionally manually UMAP.transform() new data each time). Finally, run fit_topic_model.py with a second parameter of the minimum cluster size. This should save the model in /models.
