from typing import Optional

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


# アンパサンド区切りのクエリパラメータで受け取るAPI
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None, r: str = None):
    return {"item_id": item_id, "q": q, "r": r}

# カンマ区切りのクエリパラメータを受け取るAPI
@app.get("/parse-params")
def parse_params(values: Optional[str] = None):
    if values:
        items = [v.strip() for v in values.split(",") if v.strip()]
    else:
        items = []
    return {"values": items}