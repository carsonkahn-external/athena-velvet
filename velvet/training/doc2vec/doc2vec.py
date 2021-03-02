import gensim
import json
import pandas as pd
import pickle
import spacy
import random

from datetime import datetime


# Stub for now, probably take care of pre-processing here (before NLP)
def load_data(data_path):
	df = pd.read_csv(data_path)
	return df


#TODO:// Probably move to taking a column directly
def lemmatize_column(df, save_path = None):
	"""
	Takes a pandas dataframe and 
	"""

	nlp = spacy.load('en_core_web_sm', disable=['ner',  'tok2vec', 'parser'])

	lemma_descriptions = df['description'].apply(lemmatize, nlp_pipeline=nlp)

	if save_path:
		with open(save_path, 'wb') as fh:
			pickle.dump(lemma_descriptions, fh)

	return lemma_descriptions


def lemmatize(texts, nlp_pipeline):
	"""
	Takes a string and processes through a nlp pipeline.
	Returning a string of lemmas with stop words removed.
	"""
	try:
		m = nlp_pipeline(texts)
		lemmas = " ".join([token.lemma_ for token in m if not token.is_stop])
		return lemmas
	except Exception as e:
		print(texts)
		print(e)
		return ""


def tagged_docs_from_series(docs):
	"""
		Takes a pandas series (e.g. a sigle column) and returns a 
		list of tagged documents.

		Documents are in themselves a list of lemmas.

		A tagged document has a unqiue integer id - useful for referencing later
	"""
	documents = [doc.split() for doc in docs.tolist()]

	tagged_docs = [gensim.models.doc2vec.TaggedDocument(td, [idx]) \
		for idx, td in enumerate(docs.tolist())]

	return tagged_docs


def build_model(save_path):
	doc2vec = gensim.models.doc2vec.Doc2Vec(vector_size=1000, min_count=10, epochs=20)
	doc2vec.build_vocab(tagged_docs)

	return doc2vec


def train(model):
	model.train(tagged_docs, total_examples=doc2vec.corpus_count, epochs=40)

	model.save('models/finace_doc2vec_40_epoch')


def sample_model():
	"""
		Get's a random document from our training data and returns 
		similar docs.
	"""

	doc_id = random.randint(0, len(tagged_docs) - 1)

	words = list(tagged_docs[doc_id].words)
	inferred_vector = doc2vec.infer_vector(words)

	sims = doc2vec.docvecs.most_similar([inferred_vector], topn=10)

	# Compare and print the most/median/least similar documents from the train corpus
	print('Test Document ({}): «{}»\n'.format(doc_id, ''.join(words)))
	for label, index in [('Median', 5)]:
		print(f'{label}: {tagged_docs[sims[index][0]]}')
		print(u'%s %s: «%s»\nlp' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))
