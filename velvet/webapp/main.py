from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


import gensim
import pandas as pd

app = FastAPI()
app.mount("/views/static", StaticFiles(directory="views/static"), name="static")

templates = Jinja2Templates(directory="views")

doc2vec = gensim.models.doc2vec.Doc2Vec.load('./models/finace_doc2vec_40_epoch')
df = pd.read_csv('./data/large_only_finance.csv')
print(df.shape)

# Breakout model logic to seperate file
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
        #payload[index] = row['title']
        
    
    print(payload)
    import json
    return json.dumps(payload)



@app.get("/compare", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("static/compare.html", {"request": request})


@app.get("/")
def read_root():
    #Break this into /view/.html    
    html_content = """
    <html>
        <head>
            <title>Doc2Vec</title>
        </head>
        <style>
            #user_desc {
                width: 300px;
                height: 100px;
            }
        </style>
        <body>
            <h1>Insert description to get similar works</h1>
            <textarea id='user_desc'></textarea>
            <input type='button' value='Go!' onclick='load_sim()'\>
            <div id='similar_postings'></div>
            
            <script>
            
function load_sim(){
	let user_text = document.getElementById('user_desc').value
	let similar_postings = document.getElementById('similar_postings')


	function parse_return(data){
		data = JSON.parse(data)
	    for (var key in data){
	        similar_postings.innerHTML = ''
            similar_postings.innerHTML += '<b>' + data[key]['title'] + '</b>' + '<br/>';
            similar_postings.innerHTML += data[key]['description'] + '<br/>'
	    }
	}

	fetch('/model?' + new URLSearchParams({
	    	'query_string': user_text}))
	    	.then(response => response.json())
			.then((data) => parse_return(data))
}
            </script>
        
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.get("/model")
def get_similiar_posting(query_string: str):
    return closest_post(query_string)


