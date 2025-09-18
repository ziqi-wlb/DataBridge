"""
DataLoader classes for runtime dataset loading
"""

import os
import logging
from typing import Iterator, List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
import numpy as np

from .dataset import Document, Dataset
from ..formats.registry import registry

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Base class for all data loaders"""
    
    def __init__(self, 
                 dataset_path: str,
                 format_name: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize data loader
        
        Args:
            dataset_path: Path to the dataset
            format_name: Format name (auto-detect if None)
            tokenizer_path: Path to tokenizer
            **kwargs: Additional loader-specific options
        """
        self.dataset_path = dataset_path
        self.format_name = format_name
        self.tokenizer_path = tokenizer_path
        self.kwargs = kwargs
        
        # Get format handler
        if format_name:
            self.handler = registry.get_handler(format_name)
            if not self.handler:
                raise ValueError(f"Unknown format: {format_name}")
        else:
            self.handler = registry.get_handler_by_extension(dataset_path)
            if not self.handler:
                raise ValueError(f"Cannot auto-detect format for: {dataset_path}")
        
        # Initialize handler with tokenizer
        self.handler = self.handler.__class__(tokenizer_path=tokenizer_path)
        
        logger.info(f"Initialized {self.__class__.__name__} for {self.handler.format_name} format")
    
    @abstractmethod
    def __iter__(self) -> Iterator[Any]:
        """Iterate over the dataset"""
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Get dataset length"""
        pass
    
    def get_document_count(self) -> int:
        """Get total document count"""
        return self.handler.get_document_count(self.dataset_path)


class PyTorchDataLoader(DataLoader):
    """PyTorch-compatible data loader"""
    
    def __init__(self, 
                 dataset_path: str,
                 format_name: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 num_workers: int = 0,
                 collate_fn: Optional[Callable] = None,
                 **kwargs):
        """
        Initialize PyTorch data loader
        
        Args:
            dataset_path: Path to the dataset
            format_name: Format name (auto-detect if None)
            tokenizer_path: Path to tokenizer
            batch_size: Batch size for data loading
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            collate_fn: Custom collate function
            **kwargs: Additional options
        """
        super().__init__(dataset_path, format_name, tokenizer_path, **kwargs)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn or self._default_collate_fn
        
        # Create PyTorch dataset wrapper
        self.torch_dataset = PyTorchDatasetWrapper(self.handler, self.dataset_path)
        
        # Create PyTorch DataLoader
        self.torch_dataloader = TorchDataLoader(
            self.torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
    
    def _default_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Default collate function for batching"""
        if not batch:
            return {}
        
        # Extract common keys
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [item[key] for item in batch]
            
            # Handle different data types
            if key == 'tokens' and all(isinstance(v, list) for v in values):
                # Pad token sequences
                max_len = max(len(v) for v in values)
                padded = []
                for v in values:
                    padded.append(v + [0] * (max_len - len(v)))
                collated[key] = torch.tensor(padded, dtype=torch.long)
            elif key == 'text' and all(isinstance(v, str) for v in values):
                # Keep as list of strings
                collated[key] = values
            elif key in ['id', 'length', 'token_count'] and all(isinstance(v, (int, float)) for v in values):
                # Convert to tensor
                collated[key] = torch.tensor(values, dtype=torch.long)
            else:
                # Keep as list
                collated[key] = values
        
        return collated
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches"""
        return iter(self.torch_dataloader)
    
    def __len__(self) -> int:
        """Get number of batches"""
        return len(self.torch_dataloader)
    
    def get_torch_dataloader(self) -> TorchDataLoader:
        """Get the underlying PyTorch DataLoader"""
        return self.torch_dataloader


class PyTorchDatasetWrapper(TorchDataset):
    """PyTorch Dataset wrapper for format handlers"""
    
    def __init__(self, handler: Dataset, dataset_path: str):
        """
        Initialize PyTorch dataset wrapper
        
        Args:
            handler: Format handler
            dataset_path: Path to dataset
        """
        self.handler = handler
        self.dataset_path = dataset_path
        self._documents = None
        self._load_documents()
    
    def _load_documents(self):
        """Load all documents into memory"""
        logger.info(f"Loading documents from {self.dataset_path}")
        self._documents = list(self.handler.load(self.dataset_path))
        logger.info(f"Loaded {len(self._documents)} documents")
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self._documents)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        if idx >= len(self._documents):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self._documents)}")
        
        doc = self._documents[idx]
        return doc.to_dict()


class HuggingFaceDataLoader(DataLoader):
    """HuggingFace datasets-compatible data loader"""
    
    def __init__(self, 
                 dataset_path: str,
                 format_name: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize HuggingFace data loader
        
        Args:
            dataset_path: Path to the dataset
            format_name: Format name (auto-detect if None)
            tokenizer_path: Path to tokenizer
            **kwargs: Additional options
        """
        super().__init__(dataset_path, format_name, tokenizer_path, **kwargs)
        
        # Load documents
        self._documents = list(self.handler.load(self.dataset_path))
        logger.info(f"Loaded {len(self._documents)} documents for HuggingFace format")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over documents"""
        for doc in self._documents:
            yield doc.to_dict()
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self._documents)
    
    def to_huggingface_dataset(self):
        """Convert to HuggingFace Dataset object"""
        try:
            from datasets import Dataset as HFDataset
            
            # Convert documents to list of dicts
            data = [doc.to_dict() for doc in self._documents]
            
            # Create HuggingFace dataset
            hf_dataset = HFDataset.from_list(data)
            return hf_dataset
        except ImportError:
            raise ImportError("HuggingFace datasets library not installed. Install with: pip install datasets")


class MegatronDataLoader(DataLoader):
    """Megatron-compatible data loader"""
    
    def __init__(self, 
                 dataset_path: str,
                 format_name: Optional[str] = None,
                 tokenizer_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize Megatron data loader
        
        Args:
            dataset_path: Path to the dataset
            format_name: Format name (auto-detect if None)
            tokenizer_path: Path to tokenizer
            **kwargs: Additional options
        """
        super().__init__(dataset_path, format_name, tokenizer_path, **kwargs)
        
        # For Megatron, we typically work with tokenized data
        if not self.handler.tokenizer:
            raise ValueError("Megatron data loader requires a tokenizer")
        
        # Load documents
        self._documents = list(self.handler.load(self.dataset_path))
        logger.info(f"Loaded {len(self._documents)} documents for Megatron format")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over tokenized documents"""
        for doc in self._documents:
            doc_data = doc.to_dict()
            
            # Ensure we have tokens
            if 'tokens' not in doc_data or not doc_data['tokens']:
                # Tokenize if not already done
                text = doc_data.get('text', '')
                tokens = self.handler.tokenizer.encode(text)
                doc_data['tokens'] = tokens
            
            yield doc_data
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self._documents)
    
    def get_tokenized_dataset(self):
        """Get tokenized dataset for Megatron training"""
        try:
            from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
            
            # This would create a Megatron-compatible dataset
            # Implementation depends on specific Megatron version
            logger.info("Creating Megatron-compatible tokenized dataset")
            
            # For now, return the documents with tokens
            return [doc.to_dict() for doc in self._documents]
        except ImportError:
            logger.warning("Megatron not available, returning tokenized documents")
            return [doc.to_dict() for doc in self._documents]


class DataLoaderFactory:
    """Factory for creating data loaders"""
    
    @staticmethod
    def create_loader(loader_type: str, 
                     dataset_path: str,
                     format_name: Optional[str] = None,
                     tokenizer_path: Optional[str] = None,
                     **kwargs) -> DataLoader:
        """
        Create a data loader of the specified type
        
        Args:
            loader_type: Type of loader ('pytorch', 'huggingface', 'megatron')
            dataset_path: Path to the dataset
            format_name: Format name (auto-detect if None)
            tokenizer_path: Path to tokenizer
            **kwargs: Additional options
            
        Returns:
            DataLoader instance
        """
        loader_type = loader_type.lower()
        
        if loader_type == 'pytorch':
            return PyTorchDataLoader(dataset_path, format_name, tokenizer_path, **kwargs)
        elif loader_type == 'huggingface':
            return HuggingFaceDataLoader(dataset_path, format_name, tokenizer_path, **kwargs)
        elif loader_type == 'megatron':
            return MegatronDataLoader(dataset_path, format_name, tokenizer_path, **kwargs)
        else:
            raise ValueError(f"Unknown loader type: {loader_type}. Supported types: pytorch, huggingface, megatron")
    
    @staticmethod
    def list_loader_types() -> List[str]:
        """List available loader types"""
        return ['pytorch', 'huggingface', 'megatron']


# Convenience functions
def create_pytorch_loader(dataset_path: str, 
                         format_name: Optional[str] = None,
                         tokenizer_path: Optional[str] = None,
                         **kwargs) -> PyTorchDataLoader:
    """Create a PyTorch data loader"""
    return DataLoaderFactory.create_loader('pytorch', dataset_path, format_name, tokenizer_path, **kwargs)


def create_huggingface_loader(dataset_path: str, 
                             format_name: Optional[str] = None,
                             tokenizer_path: Optional[str] = None,
                             **kwargs) -> HuggingFaceDataLoader:
    """Create a HuggingFace data loader"""
    return DataLoaderFactory.create_loader('huggingface', dataset_path, format_name, tokenizer_path, **kwargs)


def create_megatron_loader(dataset_path: str, 
                          format_name: Optional[str] = None,
                          tokenizer_path: Optional[str] = None,
                          **kwargs) -> MegatronDataLoader:
    """Create a Megatron data loader"""
    return DataLoaderFactory.create_loader('megatron', dataset_path, format_name, tokenizer_path, **kwargs)
