import gensim
import json
import pandas as pd
import spacy


# todo - we can make this dynamically load a different model
doc2vec = gensim.models.doc2vec.Doc2Vec.load('models/doc2vec/finace_doc2vec_40_epoch')
df = pd.read_csv('models/data/large_only_finance.csv')


nlp = spacy.load('en_core_web_sm', disable=['ner',  'tok2vec', 'parser'])

def lemmatize(texts):
    try:
        m = nlp(texts)
        lemmas = " ".join([token.lemma_ for token in m if not token.is_stop])
        return lemmas
    except Exception as e:
        print(texts)
        print(e)
        return ""

def doctag_to_doc(doctag: int):
    return df.iloc[doctag] 

def closest_post(query_string):
    words = query_string.split()
    
    inferred_vector = doc2vec.infer_vector(words)
    sims = doc2vec.docvecs.most_similar([inferred_vector], topn=2)
    
    payload = {}
    for index, doc in enumerate(sims):
        row = doctag_to_doc(sims[index][0])
        
        # Doing this to avoid having to write json parser for NumPy
        payload[index] = {'title': None, 'description': None}
        payload[index]['title'] = row['title']
        payload[index]['description'] = row['description']
    
    return json.dumps(payload)