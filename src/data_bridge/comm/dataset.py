"""
Dataset base class and related utilities
"""

from typing import List, Dict, Any, Iterator, Optional, Union
from abc import ABC, abstractmethod
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Document:
    """Represents a single document in the dataset"""
    
    def __init__(self, data: Dict[str, Any], doc_id: Optional[str] = None):
        """
        Initialize a document
        
        Args:
            data: Document data dictionary
            doc_id: Optional document ID
        """
        self.data = data
        self.doc_id = doc_id or data.get('id', f"doc_{id(data)}")
    
    def __getitem__(self, key: str) -> Any:
        """Get item by key"""
        return self.data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set item by key"""
        self.data[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists"""
        return key in self.data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get item with default value"""
        return self.data.get(key, default)
    
    def keys(self):
        """Get all keys"""
        return self.data.keys()
    
    def values(self):
        """Get all values"""
        return self.data.values()
    
    def items(self):
        """Get all items"""
        return self.data.items()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return self.data.copy()
    
    def __repr__(self) -> str:
        return f"Document(id={self.doc_id}, keys={list(self.data.keys())})"
    
    def __len__(self) -> int:
        return len(self.data)


class Dataset(ABC):
    """Base class for all dataset formats"""
    
    def __init__(self, tokenizer_path: Optional[str] = None):
        """
        Initialize dataset
        
        Args:
            tokenizer_path: Path to tokenizer for text processing
        """
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self._load_tokenizer()
    
    def _load_tokenizer(self) -> None:
        """Load tokenizer if path is provided"""
        if self.tokenizer_path:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
                logger.info(f"Loaded tokenizer from {self.tokenizer_path}")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer from {self.tokenizer_path}: {e}")
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Get format name"""
        pass
    
    @property
    @abstractmethod
    def file_extensions(self) -> List[str]:
        """Get supported file extensions"""
        pass
    
    @abstractmethod
    def can_load(self, path: str) -> bool:
        """
        Check if this format can load the given path
        
        Args:
            path: Path to check
            
        Returns:
            True if this format can handle the path
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> Iterator[Document]:
        """
        Load documents from path
        
        Args:
            path: Path to load from
            
        Yields:
            Document objects
        """
        pass
    
    @abstractmethod
    def save(self, documents: Union[Iterator[Document], List[Document]], 
             output_path: str, **kwargs) -> None:
        """
        Save documents to path
        
        Args:
            documents: Documents to save
            output_path: Path to save to
            **kwargs: Additional format-specific options
        """
        pass
    
    def get_document_count(self, path: str) -> Optional[int]:
        """
        Get document count if available
        
        Args:
            path: Path to check
            
        Returns:
            Document count or None if not available
        """
        return None
    
    def validate_path(self, path: str) -> bool:
        """
        Validate that path exists and is accessible
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid
        """
        try:
            path_obj = Path(path)
            return path_obj.exists() and (path_obj.is_file() or path_obj.is_dir())
        except Exception:
            return False
    
    def convert_to_documents(self, data: List[Dict[str, Any]]) -> List[Document]:
        """
        Convert list of dictionaries to Document objects
        
        Args:
            data: List of document dictionaries
            
        Returns:
            List of Document objects
        """
        return [Document(item) for item in data]
    
    def convert_from_documents(self, documents: Union[Iterator[Document], List[Document]]) -> List[Dict[str, Any]]:
        """
        Convert Document objects to list of dictionaries
        
        Args:
            documents: Document objects
            
        Returns:
            List of document dictionaries
        """
        if isinstance(documents, Iterator):
            documents = list(documents)
        return [doc.to_dict() for doc in documents]
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(format={self.format_name})"


class DatasetCollection:
    """Collection of datasets for batch operations"""
    
    def __init__(self, datasets: List[Dataset]):
        """
        Initialize dataset collection
        
        Args:
            datasets: List of dataset objects
        """
        self.datasets = datasets
        self._format_map = {ds.format_name: ds for ds in datasets}
    
    def get_dataset(self, format_name: str) -> Optional[Dataset]:
        """Get dataset by format name"""
        return self._format_map.get(format_name)
    
    def list_formats(self) -> List[str]:
        """List all available formats"""
        return list(self._format_map.keys())
    
    def convert(self, input_path: str, output_path: str, 
                input_format: Optional[str] = None,
                output_format: Optional[str] = None,
                **kwargs) -> None:
        """
        Convert between formats
        
        Args:
            input_path: Input file/directory path
            output_path: Output file/directory path
            input_format: Input format name (auto-detect if None)
            output_format: Output format name (auto-detect if None)
            **kwargs: Additional options
        """
        # Get input dataset
        if input_format:
            input_dataset = self.get_dataset(input_format)
            if not input_dataset:
                raise ValueError(f"Unknown input format: {input_format}")
        else:
            input_dataset = self._auto_detect_format(input_path)
            if not input_dataset:
                raise ValueError(f"Cannot auto-detect format for: {input_path}")
        
        # Get output dataset
        if output_format:
            output_dataset = self.get_dataset(output_format)
            if not output_dataset:
                raise ValueError(f"Unknown output format: {output_format}")
        else:
            output_dataset = self._auto_detect_format(output_path)
            if not output_dataset:
                raise ValueError(f"Cannot auto-detect format for: {output_path}")
        
        logger.info(f"Converting {input_dataset.format_name} to {output_dataset.format_name}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        
        # Load documents
        documents = input_dataset.load(input_path)
        
        # Save documents
        output_dataset.save(documents, output_path, **kwargs)
        
        logger.info("Conversion completed successfully")
    
    def _auto_detect_format(self, path: str) -> Optional[Dataset]:
        """Auto-detect format from path"""
        for dataset in self.datasets:
            if dataset.can_load(path):
                return dataset
        return None
    
    def __len__(self) -> int:
        return len(self.datasets)
    
    def __iter__(self):
        return iter(self.datasets)
    
    def __repr__(self) -> str:
        return f"DatasetCollection({len(self.datasets)} formats: {', '.join(self.list_formats())})"
