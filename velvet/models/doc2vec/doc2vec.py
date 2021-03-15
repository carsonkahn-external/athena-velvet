import gensim
import json
import os
import pandas as pd
import pickle
import random
import spacy

from datetime import datetime
from pathlib import Path


class Doc2Vec:
	def __init__(self, model=None, tagged_docs=None):
		self.model = model
		self.tagged_docs = tagged_docs

		self.nlp_pipeline = spacy.load('en_core_web_sm', disable=['ner',  'tok2vec', 'parser'])


	def load_data(self, data_path, row_limit=None):
		df = pd.read_csv(data_path, nrows=row_limit)
		self.data = df
		return self.data


	def load_tagged_docs(self, docs_path):
		with open(docs_path, 'rb') as fh:
			self.tagged_docs = pickle.load(fh)
			return self.tagged_docs


	def load_trained_model(self, model_path):
		self.model = gensim.models.doc2vec.Doc2Vec.load(model_path)
		return self.model


	def lemmatize_column(self, df, save_path = None):
		"""Takes a pandas dataframe and returns Series of lemmatized strings.
		"""		
		lemma_descriptions = df['description'].apply(self.lemmatize)

		if save_path:
			with open(timeStamped(save_path), 'wb') as fh:
				pickle.dump(lemma_descriptions, fh)

		return lemma_descriptions


	def lemmatize(self, text):
		"""Takes a string and processes through a nlp pipeline.
		Returning a string of lemmas with stop words removed.
		"""
		try:
			m = self.nlp_pipeline(text)
			lemmas = [token.lemma_ for token in m if not token.is_stop]
			return lemmas
		except Exception as e:
			print(text)
			print(e)
			return ""


	def tagged_docs_from_series(self, docs, save_path=None):
		"""Takes a pandas series (e.g. a sigle column) and returns a 
		list of tagged documents.

		Documents are in themselves a list of lemmas.
		A tagged document has a unqiue integer id - useful for referencing later.

		We also save a corresponding TaggedDocument list that has the unalterted
		text.
		"""
		documents = [self.lemmatize(doc) for doc in docs.tolist()]

		#Perhaps move to one loop for insurance that tags line up
		tagged_parsed_docs = [gensim.models.doc2vec.TaggedDocument(td, [idx]) \
			for idx, td in enumerate(documents)]

		tagged_full_docs = [gensim.models.doc2vec.TaggedDocument(td, [idx]) \
			for idx, td in enumerate(docs)]

		if save_path:
			with open(timeStamped(save_path), 'wb') as fh:
				pickle.dump(tagged_parsed_docs, fh)

			with open(timeStamped(save_path+'_full_texts'), 'wb') as fh:
				pickle.dump(tagged_full_docs, fh)


		return tagged_parsed_docs


	def build_model(self, 
			model_params={'vector_size': 1000, 'min_count': 10, 'epochs': 20}):
		"""Builds a Doc2Vec model from supplied parameters and vocab (TaggedDocument)

		Args:
			tagged_docs: TaggedDocument object
			model_params: dict, refer to Gensim Doc2Vec for list.
		"""
		doc2vec = gensim.models.doc2vec.Doc2Vec(**model_params)
		doc2vec.build_vocab(self.tagged_docs)

		self.model = doc2vec

		return doc2vec


	def train(save_path):
		"""Some notes:
			dm is analogous to Word2Vec CBOW - i.e. NN is trained on predicting
			center context word. Other wise, SG is used - i.e. the whole document is
			used to predict a sample word
		"""
		self.model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

		self.model.save(timeStamped(save_path)+f'_{model.epochs}_epochs')

		return model


	def sample_model(self):
		"""Get's a random document from our training data and returns 
		similar docs.
		"""

		doc_id = random.randint(0, len(self.tagged_docs) - 1)

		words = self.tagged_docs[doc_id].words
		words = self.lemmatize(words)

		inferred_vector = self.model.infer_vector(words)
		potential = self.model.dv.most_similar([inferred_vector], topn=1)
		
		#We're filtering out high matches as they're likely
		#just the same document
		sims = []
		for doc in potential:
			if doc[1] < .9:
				sims.append(doc)

			if len(sims) > 3:
				break

		print(f'sims: {sims}')
		print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(words)))
		for doc in sims:
			index = doc[0]
			print(f'DOCUMENT {index}: \n \
				{self.tagged_docs[index].words} \n \n *********************************************')


	def predict_word(self, sentence, word_count=5):
		"""Takes a sentence with a missword and returns the model's
		best guess as to which word it is.

		Args:
			sentence: A string where the missing word is repalced with "?"
			word_count: The number of guesses to return
		"""
		words = self.lemmatize(sentence)
		
		#predict_output_word predicts the central word, so we need
		#to reformat our sentence to match this
		index = words.index("?")
		
		window_size = 5
		left_window = []
		right_window = []
		#Iterater through string, get up to window_size words on each side
		for i, word in enumerate(words):
			distance = index - i
			if abs(distance) < window_size:
				if distance > 0:
					left_window.append(word)
				elif distance < 0:
					right_window.append(word)

		#I wrote the above to test different padding schemes
		#Those can be experimented here, but I've found that 
		#if there's not enough context, (i.e. ? at end of single sentence)
		#it doesn't make a big difference.

		centered_sentence = left_window + right_window
		return self.model.predict_output_word(centered_sentence, topn=word_count)


	def similar_document(self, doc):
	    """Returns similar documents to the provided doc.

		Args:
			doc: A string representing a document
		"""
	    doc = self.lemmatize(doc)

	    doc_vec = self.model.infer_vector(doc)
	    sim_docs = self.model.dv.most_similar([doc_vec], topn=3)

	    full_docs = []
	    for doc in sim_docs:
	    	index = doc[0]
	    	full_docs.append(self.tagged_docs[index].words)

	    return full_docs


	@staticmethod
	def timeStamped(save_path, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
	    """Helper function to datestamp file names
	    """
	    save_path = Path(save_path)
	    fname = save_path.name
	    fname = datetime.now().strftime(fmt).format(fname=fname)

	    return os.path.join(save_path.parent, fname)