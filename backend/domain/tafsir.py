# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class TafsirRecord(BaseModel):
    """
    Tafsir commentary schema for Quranic verses (e.g. Ibn Kathir, Saadi, Jalalayn).
    """
    id: str = Field(..., description="e.g. tafsir_ibnkathir_2_255")
    mufassir: str = Field(..., description="Mufassir name e.g. ابن كثير")
    surah_number: int
    ayah_number: int
    text: str = Field(..., description="The commentary text")
    source: str = Field(default="tafsir", description="Data source type")


class TranslationRecord(BaseModel):
    """
    Quran verse translation schema (e.g. Sahih International, Pickthall, Yusuf Ali).
    """
    id: str = Field(..., description="e.g. trans_en_2_255")
    language: str = Field(default="en", description="ISO 639-1 language code")
    translator: str = Field(..., description="e.g. Saheeh International")
    surah_number: int
    ayah_number: int
    text: str = Field(..., description="Translated text")
