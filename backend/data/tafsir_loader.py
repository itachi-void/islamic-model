# -*- coding: utf-8 -*-
import json
import os
from typing import List, Optional
from backend.domain.tafsir import TafsirRecord, TranslationRecord

TAFSIR_DATA_DIR = r"d:\model\data\tafsir"


class TafsirLoader:
    def __init__(self, data_dir: str = TAFSIR_DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def load_tafsir(self, mufassir: str = "ibnkathir") -> List[TafsirRecord]:
        file_path = os.path.join(self.data_dir, f"{mufassir}.json")
        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        return [TafsirRecord(**item) for item in raw]

    def get_ayah_tafsir(self, surah: int, ayah: int, mufassir: str = "ibnkathir") -> Optional[TafsirRecord]:
        all_records = self.load_tafsir(mufassir)
        for r in all_records:
            if r.surah_number == surah and r.ayah_number == ayah:
                return r
        return None
