"""
Megatron bin/idx format handler
"""

import os
import json
import logging
from typing import List, Dict, Any, Iterator
from tqdm import tqdm

from .base import BaseFormatHandler
from ..comm.dataset import Document

logger = logging.getLogger(__name__)


class BinIdxFormatHandler(BaseFormatHandler):
    """Handler for Megatron bin/idx format"""
    
    @property
    def format_name(self) -> str:
        return "binidx"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".bin", ".idx"]
    
    def can_load(self, path: str) -> bool:
        """Check if this handler can load the given path"""
        # Check if both .bin and .idx files exist
        bin_file = f"{path}.bin"
        idx_file = f"{path}.idx"
        
        return os.path.exists(bin_file) and os.path.exists(idx_file)
    
    def load(self, path: str) -> Iterator[Document]:
        """Load documents from bin/idx dataset"""
        logger.info(f"Loading bin/idx dataset from {path}")
        logger.info(f"Tokenizer path: {self.tokenizer_path}")
        logger.info(f"Tokenizer loaded: {self.tokenizer is not None}")
        
        # Load the indexed dataset
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        dataset = IndexedDataset(path, multimodal=False, mmap=True)
        
        num_documents = len(dataset)
        logger.info(f"Loaded dataset with {num_documents} documents")
        
        # Iterate through documents
        for doc_idx in tqdm(range(num_documents), desc="Loading documents"):
            # Get document token IDs
            token_ids = dataset.get(doc_idx)
            
            # Convert to text if tokenizer is available
            if self.tokenizer:
                text = self.tokenizer.decode(token_ids)
            else:
                text = f"[TOKENS: {len(token_ids)} tokens]"
            
            # Create minimal document data (avoid unnecessary fields for performance)
            doc_data = {'id': doc_idx, 'text': text}
            yield Document(doc_data, doc_id=str(doc_idx))

        logger.info(f"Finished loading bin/idx dataset from {path}")
    
    def save(self, documents: Iterator[Document], output_path: str, **kwargs) -> None:
        """Save documents to bin/idx format"""
        # Note: Writing bin/idx format is complex and typically done by Megatron tools
        # For now, we'll save as JSONL and let the user know
        jsonl_path = f"{output_path}.jsonl"
        
        logger.warning("Bin/idx format writing is not implemented. Saving as JSONL instead.")
        logger.info(f"Saving to JSONL: {jsonl_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for doc in tqdm(documents, desc="Writing JSONL"):
                f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {jsonl_path}. Use Megatron tools to convert to bin/idx format.")
    
    def get_document_count(self, path: str) -> int:
        """Get the number of documents in bin/idx dataset"""
        from megatron.core.datasets.indexed_dataset import IndexedDataset
        dataset = IndexedDataset(path, multimodal=False, mmap=True)
        return len(dataset)
