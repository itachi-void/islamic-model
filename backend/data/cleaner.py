# -*- coding: utf-8 -*-
import re

class DataCleaner:
    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by removing extra whitespaces, newlines,
        and other standard artifacts.
        """
        # Collapse multiple whitespaces/tabs/newlines into a single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
