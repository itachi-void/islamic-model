# -*- coding: utf-8 -*-
from pydantic import BaseModel
from typing import List
from backend.domain.document import BaseDocument

class SearchResponse(BaseModel):
    query: str
    count: int
    documents: List[BaseDocument]
