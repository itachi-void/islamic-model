# -*- coding: utf-8 -*-
from typing import List
from backend.domain.document import BaseDocument

class DocumentChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: BaseDocument) -> List[BaseDocument]:
        """
        Splits a single document into smaller text chunks, returning a list of BaseDocument instances.
        Each chunk inherits properties from the parent but has a distinct ID (e.g. parent_id_chunk_index).
        """
        text = document.text
        if len(text) <= self.chunk_size:
            return [document]

        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Construct a sub-document for this chunk
            chunk_doc = BaseDocument(
                id=f"{document.id}_chunk_{chunk_idx}",
                type=document.type,
                source=document.source,
                text=chunk_text,
                metadata={
                    **document.metadata,
                    "chunk_index": chunk_idx,
                    "parent_id": document.id
                }
            )
            chunks.append(chunk_doc)
            
            start += (self.chunk_size - self.chunk_overlap)
            chunk_idx += 1
            
        return chunks
