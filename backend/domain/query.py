# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class SearchQuery(BaseModel):
    text: str = Field(..., description="The query string to search for")
    limit: int = Field(default=5, ge=1, le=50, description="The max number of search results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata key-value filters to apply")
