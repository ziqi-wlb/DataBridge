"""
JSONL format handler
"""

import os
import json
import logging
from typing import List, Dict, Any, Iterator
from tqdm import tqdm

from .base import BaseFormatHandler
from ..comm.dataset import Document

logger = logging.getLogger(__name__)


class JsonlFormatHandler(BaseFormatHandler):
    """Handler for JSONL format"""
    
    @property
    def format_name(self) -> str:
        return "jsonl"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".jsonl", ".json"]
    
    def can_load(self, path: str) -> bool:
        """Check if this handler can load the given path"""
        if not self.validate_path(path):
            return False
        
        # Check if it's a JSONL file
        _, ext = os.path.splitext(path)
        return ext.lower() in self.file_extensions
    
    def load(self, path: str) -> Iterator[Document]:
        """Load documents from JSONL file"""
        logger.info(f"Loading JSONL from {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(tqdm(f, desc="Loading JSONL"), 1):
                line = line.strip()
                if not line:
                    continue
                
                doc_data = json.loads(line)
                # Ensure document has an ID
                if 'id' not in doc_data:
                    doc_data['id'] = line_num - 1
                
                yield Document(doc_data)
        
        logger.info(f"Finished loading JSONL from {path}")
    
    def save(self, documents: Iterator[Document], output_path: str, **kwargs) -> None:
        """Save documents to JSONL file"""
        logger.info(f"Saving JSONL to {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in tqdm(documents, desc="Writing JSONL"):
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Finished saving JSONL to {output_path}")
    
    def get_document_count(self, path: str) -> int:
        """Get the number of documents in JSONL file"""
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
