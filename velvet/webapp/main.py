import json
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from models import doc2vec

app = FastAPI()
app.mount("/views/static", StaticFiles(directory="views/static"), name="static")

templates = Jinja2Templates(directory="views")

model = doc2vec.load_trained_model('./models/doc2vec/5_test/2021-03-10-19-31-13_finance_doc2vec_5_epochs')
tagged_docs = doc2vec.load_tagged_docs('./models/doc2vec/finance_tagged_docs_full_text.pickle')


@app.get("/compare", response_class=HTMLResponse)
async def get_compare(request: Request):
    return templates.TemplateResponse("static/compare.html", {"request": request})


@app.get("/get_similiar_posting")
def get_similiar_posting(query_string: str):
	sim = doc2vec.similar_document(model, tagged_docs, query_string)
	return  json.dumps(sim)


@app.get("/get_missing_word")
def get_missing_word(query_string: str):
	sim = doc2vec.similar_document(model, tagged_docs, query_string)
	return  json.dumps(sim)
