from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from models import doc2vec

app = FastAPI()
app.mount("/views/static", StaticFiles(directory="views/static"), name="static")

templates = Jinja2Templates(directory="views")

@app.get("/compare", response_class=HTMLResponse)
async def get_compare(request: Request):
    return templates.TemplateResponse("static/compare.html", {"request": request})

@app.get("/model")
def get_similiar_posting(query_string: str):
    return doc2vec.closest_post(query_string)


