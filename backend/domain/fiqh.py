# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class MadhhabDetail(BaseModel):
    """Specific ruling and evidence for one of the 4 Sunni Madhhabs."""
    school: str = Field(..., description="hanafi, maliki, shafii, or hanbali")
    ruling: str = Field(..., description="e.g. واجب, مستحب, مكروه, جائز")
    evidence: str = Field(..., description="Primary textual evidence cited by school")
    book_source: Optional[str] = Field(None, description="e.g. المغني لابن قدامة, الهداية للبرهان المرغيناني")


class MadhhabRuling(BaseModel):
    """
    Standardized Fiqh Comparative Schema across the 4 Sunni Jurisprudential Schools:
    Hanafi, Maliki, Shafi'i, and Hanbali.
    """
    id: str = Field(..., description="e.g. fiqh_tayammum_01")
    topic: str = Field(..., description="The jurisprudence topic e.g. حكم التيمم عند عدم الماء")
    fatwa: str = Field(..., description="Concise summary ruling")
    
    hanafi: Optional[MadhhabDetail] = None
    maliki: Optional[MadhhabDetail] = None
    shafii: Optional[MadhhabDetail] = None
    hanbali: Optional[MadhhabDetail] = None
    
    differences: List[str] = Field(default_factory=list, description="Key points of divergence among schools")
    evidences: List[Dict[str, Any]] = Field(default_factory=list, description="Unified Quran/Sunnah evidence list")
    sources: List[str] = Field(default_factory=list, description="Primary classical Fiqh references")
