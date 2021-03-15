import json
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from models.doc2vec.doc2vec import Doc2Vec

app = FastAPI()

templates = Jinja2Templates(directory="./webapp/views")

doc2vec = Doc2Vec()
model = doc2vec.load_trained_model('./models/trained_models/40_test/2021-03-10-20-21-04_finance_doc2vec_40_epochs')
tagged_docs = doc2vec.load_tagged_docs('./models/trained_models/2021-03-12-23-17-38_finance_tagged_docs.pickle_full_texts')


@app.get("/compare", response_class=HTMLResponse)
async def get_compare(request: Request):
    return templates.TemplateResponse("/static/compare.html", {"request": request})


@app.get("/get_similiar_posting")
def get_similiar_posting(query_string: str):
	sim = doc2vec.similar_document(query_string)
	return  json.dumps(sim)


@app.get("/get_missing_word")
def get_missing_word(query_string: str):
	words = doc2vec.predict_word(query_string, 5)
	words = [[t[0], t[1].item()] for t in words]
	print(words)
	return json.dumps(words)
