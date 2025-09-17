"""
WebDataset format handler
"""

import os
import json
import logging
from typing import List, Dict, Any, Iterator
from tqdm import tqdm
import webdataset as wds

from .base import BaseFormatHandler
from ..comm.dataset import Document

logger = logging.getLogger(__name__)


class WebDatasetFormatHandler(BaseFormatHandler):
    """Handler for WebDataset format"""
    
    @property
    def format_name(self) -> str:
        return "webdataset"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".tar"]
    
    def can_load(self, path: str) -> bool:
        """Check if this handler can load the given path"""
        if not self.validate_path(path):
            return False
        
        # Check if it's a directory with WebDataset files
        if not os.path.isdir(path):
            return False
        
        # Look for shard files or dataset info
        files = os.listdir(path)
        has_shards = any(f.startswith('shard_') and f.endswith('.tar') for f in files)
        has_info = 'dataset_info.json' in files
        
        return has_shards or has_info
    
    def load(self, path: str) -> Iterator[Document]:
        """Load documents from WebDataset"""
        logger.info(f"Loading WebDataset from {path}")
        
        # Find all shard files
        shard_files = []
        for file in os.listdir(path):
            if file.startswith('shard_') and file.endswith('.tar'):
                shard_files.append(os.path.join(path, file))
        
        shard_files.sort()
        logger.info(f"Found {len(shard_files)} shard files")
        
        # Load dataset info if available
        info_file = os.path.join(path, 'dataset_info.json')
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                dataset_info = json.load(f)
            logger.info(f"Dataset info: {dataset_info}")
        
        # Iterate through shards
        for shard_file in tqdm(shard_files, desc="Loading shards"):
            dataset = wds.WebDataset(shard_file)
            for sample in dataset:
                # Extract text and metadata
                text = sample['text'].decode('utf-8') if isinstance(sample['text'], bytes) else sample['text']
                
                # Extract metadata
                metadata = {}
                if 'json' in sample:
                    metadata = json.loads(sample['json'].decode('utf-8') if isinstance(sample['json'], bytes) else sample['json'])
                
                # Extract tokens if available
                tokens = []
                if 'tokens.json' in sample:
                    tokens = json.loads(sample['tokens.json'].decode('utf-8') if isinstance(sample['tokens.json'], bytes) else sample['tokens.json'])
                
                # Extract ID from __key__ if no json metadata available
                doc_id = metadata.get('id', 'unknown')
                if doc_id == 'unknown' and '__key__' in sample:
                    # Extract numeric ID from __key__ format like "doc_00000000"
                    key = sample['__key__']
                    if key.startswith('doc_'):
                        try:
                            doc_id = int(key[4:])  # Remove "doc_" prefix and convert to int
                        except ValueError:
                            doc_id = 0  # Fallback to 0 if conversion fails
                
                doc_data = {
                    'id': doc_id,
                    'text': text,
                    'tokens': tokens,
                    'length': metadata.get('length', len(text)),
                    'token_count': metadata.get('token_count', len(tokens))
                }
                yield Document(doc_data)
        
        logger.info(f"Finished loading WebDataset from {path}")
    
    def save(self, documents: Iterator[Document], output_path: str, **kwargs) -> None:
        """Save documents to WebDataset format"""
        shard_size = kwargs.get('shard_size', 1000)
        
        logger.info(f"Saving WebDataset to {output_path}")
        logger.info(f"Shard size: {shard_size}")
        
        # Ensure output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Convert iterator to list for counting
        doc_list = list(documents)
        num_documents = len(doc_list)
        num_shards = (num_documents + shard_size - 1) // shard_size
        
        logger.info(f"Creating {num_shards} shards with {shard_size} documents per shard")
        
        # Create shards
        with wds.ShardWriter(
            os.path.join(output_path, "shard_%06d.tar"), 
            maxcount=shard_size
        ) as sink:
            for doc in tqdm(doc_list, desc="Writing shards"):
                doc_data = doc.to_dict()
                doc_id = doc_data.get('id', 0)
                text = doc_data.get('text', '')
                
                # Create sample (matching standard JSONL to WebDataset format)
                sample = {
                    "__key__": f"doc_{int(doc_id):08d}",
                    "text": text.encode('utf-8')
                }

                sink.write(sample)
        
        # Create dataset info file
        self._create_dataset_info(output_path, num_documents, num_shards, shard_size)
        
        logger.info(f"Finished saving WebDataset to {output_path}")
    
    def _create_dataset_info(self, output_dir: str, num_documents: int, num_shards: int, shard_size: int) -> None:
        """Create dataset info file"""
        info_file = os.path.join(output_dir, "dataset_info.json")
        dataset_info = {
            "num_documents": num_documents,
            "num_shards": num_shards,
            "shard_size": shard_size,
            "format": "webdataset",
            "created_by": "DataBridge",
            "version": "0.1.0"
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created dataset info file: {info_file}")
