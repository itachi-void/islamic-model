# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class KnowledgeCitation(BaseModel):
    """
    Unified Polymorphic Citation Envelope across all Islamic Knowledge Sources:
    Quran, Hadith, Tafsir, Fiqh, Seerah, Aqeedah.
    """
    id: str = Field(..., description="Unique document ID e.g. 2:255, bukhari_52")
    source: str = Field(..., description="Legacy source name e.g. quran, bukhari")
    source_type: str = Field(..., description="e.g. quran, hadith, tafsir, fiqh")
    title: str = Field(..., description="e.g. سورة البقرة, صحيح البخاري")
    reference: str = Field(..., description="e.g. 2:255, Bukhari 52")
    text: str
    surah: Optional[str] = Field(None, description="Quran Surah Name if source_type is quran")
    ayah: Optional[int] = Field(None, description="Quran Ayah Number if source_type is quran")
    confidence: float = Field(default=1.0)
    details: Dict[str, Any] = Field(default_factory=dict, description="Source-specific metadata e.g. narrator, grade, mufassir, madhhab")
