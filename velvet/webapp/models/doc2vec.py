import gensim
import json
import os
import pandas as pd
import pickle
import spacy
import random

from datetime import datetime
from pathlib import Path

# Todo:// Can make this a class - would avoid having to 
# pass same vars each time


# Stub for now, probably take care of pre-processing here (before NLP)
def load_data(data_path, row_limit=None):
	df = pd.read_csv(data_path, nrows=row_limit)
	return df


def load_trained_model(model_path):
	return gensim.models.doc2vec.Doc2Vec.load(model_path)


def load_tagged_docs(docs_path):
	with open(docs_path, 'rb') as fh:
		return pickle.load(fh)


#TODO:// Probably move to taking a column directly
def lemmatize_column(df, save_path = None):
	"""
	Takes a pandas dataframe and returns Series of lemmatized strings.
	"""

	nlp = spacy.load('en_core_web_sm', disable=['ner',  'tok2vec', 'parser'])
	
	lemma_descriptions = df['description'].apply(lemmatize, nlp_pipeline=nlp)

	if save_path:
		with open(timeStamped(save_path), 'wb') as fh:
			pickle.dump(lemma_descriptions, fh)

	return lemma_descriptions


def lemmatize(text, nlp_pipeline):
	"""
	Takes a string and processes through a nlp pipeline.
	Returning a string of lemmas with stop words removed.
	"""
	try:
		m = nlp_pipeline(text)
		lemmas = [token.lemma_ for token in m if not token.is_stop]
		return lemmas
	except Exception as e:
		print(text)
		print(e)
		return ""


def tagged_docs_from_series(docs, save_path=None):
	"""
		Takes a pandas series (e.g. a sigle column) and returns a 
		list of tagged documents.

		Documents are in themselves a list of lemmas.

		A tagged document has a unqiue integer id - useful for referencing later
	"""
	nlp = spacy.load('en_core_web_sm', disable=['ner',  'tok2vec', 'parser'])

	documents = [lemmatize(doc, nlp) for doc in docs.tolist()]

	tagged_docs = [gensim.models.doc2vec.TaggedDocument(td, [idx]) \
		for idx, td in enumerate(documents)]

	if save_path:
		with open(timeStamped(save_path), 'wb') as fh:
			pickle.dump(tagged_docs, fh)


	return tagged_docs


def build_model(tagged_docs, model_params={'vector_size': 1000, 'min_count': 10, 'epochs': 20}):
	doc2vec = gensim.models.doc2vec.Doc2Vec(**model_params)
	doc2vec.build_vocab(tagged_docs)

	return doc2vec


def train(save_path, model, tagged_docs):
	"""
	Some notes:
		dm is analogous to Word2Vec CBOW - i.e. NN is trained on predicting
		center context word. Other wise, SG is used - i.e. the whole document is
		used to predict a sample word
	"""
	model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

	model.save(timeStamped(save_path)+f'_{model.epochs}_epochs')

	return model


def sample_model(model, tagged_docs):
	"""
	Get's a random document from our training data and returns 
	similar docs.
	"""

	doc_id = random.randint(0, len(tagged_docs) - 1)

	words = tagged_docs[doc_id].words
	#print(words)

	inferred_vector = model.infer_vector(words)

	potential = model.dv.most_similar([inferred_vector], topn=10)
	
	sims = []
	for doc in potential:
		if doc[1] < .9:
			sims.append(doc)

		if len(sims) > 3:
			break

	print(f'sims: {sims}')
	# Compare and print the most/median/least similar documents from the train corpus
	print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(words)))
	for doc in sims:
		index = doc[0]
		print(f'DOCUMENT {index}: \n {" ".join(tagged_docs[index].words)} \n \n *********************************************')
		#print(u'%s %s: «%s»\nlp' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))


def predict_word(model, sentence, word_count):
	nlp = spacy.load('en_core_web_sm', disable=['ner',  'tok2vec', 'parser'])
	sentence = lemmatize(sentence, nlp)
	print(sentence)

	return model.predict_output_word(sentence, topn=word_count)


def similar_document(model, tagged_docs, doc):
    nlp = spacy.load('en_core_web_sm', disable=['ner',  'tok2vec', 'parser'])
    doc = lemmatize(doc, nlp)

    doc_vec = model.infer_vector(doc)
    sim_docs = model.dv.most_similar([doc_vec], topn=3)

    full_docs = []
    for doc in sim_docs:
    	index = doc[0]
    	full_docs.append(tagged_docs[index].words)

    return full_docs


### Utils

def timeStamped(save_path, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """
    Helper function to datestamp file names
    """
    save_path = Path(save_path)
    fname = save_path.name
    fname = datetime.now().strftime(fmt).format(fname=fname)

    return os.path.join(save_path.parent, fname)

# TODO:// File management