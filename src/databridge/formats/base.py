"""
Base format handler for dataset formats
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Iterator

from ..comm.dataset import Dataset, Document

logger = logging.getLogger(__name__)


class BaseFormatHandler(Dataset):
    """Base class for all dataset format handlers"""
    
    @abstractmethod
    def load(self, path: str) -> Iterator[Document]:
        """
        Load documents from the given path
        
        Args:
            path: Path to load from
            
        Yields:
            Document objects
        """
        pass
    
    @abstractmethod
    def save(self, documents: Iterator[Document], output_path: str, **kwargs) -> None:
        """
        Save documents to the given path
        
        Args:
            documents: Iterator of Document objects
            output_path: Path to save to
            **kwargs: Additional save parameters
        """
        pass