# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class BaseDocument(BaseModel):
    id: str
    type: str = Field(..., description="e.g. verse, hadith, tafsir")
    source: str = Field(..., description="e.g. quran, bukhari")
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
