from doc2vec import *
import pandas as pd
import pickle


def setup_from_saved_lemmas(lemma_path):
	"""
	Load a pickled file of a pandas series where each
	entry is a document that has been lemmatized.
	"""
	try:
		with open(lemma_path, 'rb') as fh:
			lemma_descriptions = pickle.load(fh)

		return lemma_descriptions

	except Exception as e:
		print(e)


def build_toy_set(data_path, count):
	"""
	Returns a small lemmatized series of documents,
	useful for debugging.
	"""
	df = pd.read_csv(data_path, nrows=count)
	descriptions = lemmatize_column(df, f'../checkpoints/{count}_lemma_descriptions.pickle')
	return descriptions


if __name__ == '__main__':
	lemmas = build_toy_set('../../data/csv/large_finance_only_postings.csv', count=1000)
	tagged_docs = tagged_docs_from_series(lemmas)

	