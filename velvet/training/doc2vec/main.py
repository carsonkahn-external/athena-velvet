import doc2vec as dv
import gensim
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


def build_toy_set(data_path, count, save = False):
	"""
	Returns a small lemmatized series of documents,
	useful for debugging.
	"""
	df = pd.read_csv(data_path, nrows=count)

	if save:
		descriptions = dv.lemmatize_column(df, f'../checkpoints/{count}_lemma_descriptions.pickle')
	else:
		descriptions = dv.lemmatize_column(df)

	return descriptions


def train_from_nothing():
	"""
	Build doc2vec from scratch
	"""
	lemmas = dv.build_toy_set('../../data/csv/large_finance_only_postings.csv', count=1000)
	tagged_docs = dv.tagged_docs_from_series(lemmas, save_path='../checkpoints/large_finance_only_postings_tagged_docs.pickle')

	model = dv.build_model(tagged_docs)
	model = dv.train('../trained_models/finance_doc2vec', model, tagged_docs)

	#return model, tagged_docs


def untrained_model():
	model = dv.build_model(tagged_docs)

	params = {'vector_size': 1000, 'min_count': 10, 'epochs': 40}
	model_2 = dv.build_model(tagged_docs, params)

	model_2 = dv.train('../checkpoints/finance_doc2vec', model_2, tagged_docs)


def load_model_and_docs(load_path):
	print("Load trained models")
	tagged_docs = dv.load_tagged_docs('../checkpoints/2021-03-05-21-13-58_large_finance_only_postings_tagged_docs.pickle')

	model = dv.load_trained_model(load_path)

	return model, tagged_docs


if __name__ == '__main__':
	model, tagged_docs = load_model_and_docs('../trained_models/2021-03-05-22-01-08_finance_doc2vec_20_epochs')

	sentence = "Goldman Sachs is seeking a risk analysis"
	predicted_word = dv.predict_word(model, sentence, 10)
	print(predicted_word)
