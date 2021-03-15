from doc2vec import Doc2Vec
import gensim
import pandas as pd
import pickle
import random


def setup_from_saved_lemmas(lemma_path):
	"""Load a pickled file of a pandas series where each
	entry is a document that has been lemmatized.
	"""
	try:
		with open(lemma_path, 'rb') as fh:
			lemma_descriptions = pickle.load(fh)

		return lemma_descriptions

	except Exception as e:
		print(e)


def build_toy_set(data_path, count, save = False):
	"""Returns a small lemmatized series of documents,
	useful for debugging.
	"""
	df = pd.read_csv(data_path, nrows=count)

	if save:
		descriptions = dv.lemmatize_column(df, f'../checkpoints/{count}_lemma_descriptions.pickle')
	else:
		descriptions = dv.lemmatize_column(df)

	return descriptions


def train_from_nothing(data_path, row_limit=None):
	"""Build doc2vec from scratch"""
	doc2vec = Doc2Vec()
	documents = doc2vec.load_data(data_path, row_limit)['description']

	tagged_docs = doc2vec.tagged_docs_from_series(documents, 
		save_path='../checkpoints/finance_tagged_docs.pickle')

	model = doc2vec.build_model(tagged_docs)
	model = doc2vec.train('../trained_models/finance_doc2vec', model, tagged_docs)

	return doc2vec


def compare_word_vectors(word_models):
	def pick_random_word(model, threshold=10):
	    while True:
	        word = random.choice(model.wv.index_to_key)
	        if model.wv.get_vecattr(word, "count") > threshold:
	            return word

	target_word = pick_random_word(word_models[0])

	for model in word_models:
	    print(f'target_word: {repr(target_word)} model: {model} similar words:')
	    for i, (word, sim) in enumerate(model.wv.most_similar(target_word, topn=10), 1):
	        print(f'    {i}. {sim:.2f} {repr(word)}')
	    print()


def trained_model(docs_path, model_path):
	doc2vec = Doc2Vec()
	doc2vec.load_tagged_docs(docs_path)
	doc2vec.load_trained_model(model_path)

	return doc2vec


if __name__ == '__main__':
	docs_path = '../trained_models/2021-03-12-23-17-38_finance_tagged_docs.pickle_full_texts'
	model_path = '../trained_models/40_test/2021-03-10-20-21-04_finance_doc2vec_40_epochs'
	model = trained_model(docs_path, model_path)