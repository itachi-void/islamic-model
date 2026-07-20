# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class HadithRecord(BaseModel):
    """
    Structured Sahih Hadith Schema for Hadith Engine Collections:
    bukhari, muslim, abudawud, tirmidhi, nasai, ibnmajah.
    """
    id: str = Field(..., description="e.g. bukhari_52")
    source: str = Field(default="bukhari", description="e.g. bukhari, muslim")
    book: str = Field(..., description="e.g. كتاب الإيمان")
    book_number: Optional[int] = None
    chapter: str = Field(..., description="e.g. باب أمور الإيمان")
    chapter_number: Optional[int] = None
    hadith_number: int = Field(..., description="e.g. 52")
    narrator: str = Field(..., description="e.g. أبي هريرة رضي الله عنه")
    matn: str = Field(..., description="The main text of the Hadith")
    sanad: Optional[str] = Field(None, description="The chain of narrators")
    grade: str = Field(default="Sahih", description="Sahih, Hasan, etc.")
    topics: List[str] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
