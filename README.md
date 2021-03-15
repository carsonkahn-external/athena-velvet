### Overview
This repo contains 3 important folders:
1. data - where raw data is stored 
2. models - a *module* that contains the Doc2Vec class and trained models
3. webapp - a simple web interface for exposing endpoints to trained models

### Doc2Vec
This class contains the infrastructure needed to build, train, save, and use the [gensim implementation](https://radimrehurek.com/gensim/models/doc2vec.html) of doc2vec.

The basic workflow consists of:
- Loading a pandas dataframe
	-	`load_data(self, data_path, row_limit=None)` where `data_path` is a relative path to a CSV, and `row_limit` is an optional parameter for how many rows from that CSV you'd like to load. 
- Passing a column to be parsed by a spaCy pipeline
	- A Doc2Vec instance has an instance variable `self.nlp_pipeline` that is used to clean up documents for processing. A default, bare bones, pipeline is loaded on instantiation but can be replaced before parsing with a custom pipeline.  
- Converting the parsed series into a tagged document
	- `tagged_docs_from_series(docs, save_path=None)` where `docs` is the series of documents as strings and if `save_path` is supplied (recommended as the process of parsing the documents can take awhile) a pickled representation is written.
	- gensim uses the concept of [TaggedDocument](https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.TaggedDocument) for storing data to be used in model training and inference. It is simply an array of tuples wherein each tuple contains `words` and `tags`. In our case we simply used a incrementing integer for the tags as no concept of shared classes was needed. 
	- One thing of note, is that for training purposes we clean and split the documents into lists of individual words. This becomes a problem at inference time as we need to reconstruct the original document for presentation. To get around this `tagged_docs_from_series` creates two TaggedDocument objects - one that represents the documents as lists of words, and the other as the original string. 
-  Specifying the hyperparameters of the model
	-  `build_model(self, model_params)` where `model_params` are all your fun dials and knobs. Take a look here for the full documentation of what you can specify.
	-  The default is as follows: `{'vector_size': 1000, 'min_count': 10, 'epochs': 20}`
-  Training the model
	-  `train(save_path)` does what it says on the tin - just follows what was specified in `build_model`

Take a look at main.py in doc2vec for more inspiration. 

### Webapp
The webapp uses [FastAPI](https://fastapi.tiangolo.com/) to expose endpoints to the Doc2Vec.

It can be started locally by running: `uvicorn webapp:app --reload` from the root directory(!)

You can then navigate to `127.0.0.0:80000/compare`