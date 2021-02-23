from typing import Optional

import gensim

doc2vec = gensim.models.doc2vec.Doc2Vec.load('./models/finace_doc2vec_40_epoch')

def closest_post(query_string):
    query_string = "This is the query string"
    words = query_string.split()
    print(words)
    #inferred_vector = doc2vec.infer_vector(words)
    #sims = doc2vec.docvecs.most_similar([inferred_vector], topn=10)
    
    return "Hello world!"

print("Called!")
closest_post("This is")
