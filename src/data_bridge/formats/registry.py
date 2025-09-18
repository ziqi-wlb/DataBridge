"""
Format registry for managing dataset format handlers
"""

import logging
import os
from typing import Dict, List, Optional, Type, Any
from .base import BaseFormatHandler
from ..comm.dataset import Dataset, Document

logger = logging.getLogger(__name__)


class FormatRegistry:
    """Registry for managing dataset format handlers"""
    
    def __init__(self):
        self._handlers: Dict[str, Type[BaseFormatHandler]] = {}
        self._extensions: Dict[str, str] = {}
    
    def register(self, handler_class: Type[BaseFormatHandler]) -> None:
        """
        Register a format handler
        
        Args:
            handler_class: Format handler class to register
        """
        # Create a temporary instance to get format info
        temp_instance = handler_class()
        format_name = temp_instance.format_name
        
        self._handlers[format_name] = handler_class
        
        # Register file extensions
        for ext in temp_instance.file_extensions:
            self._extensions[ext.lower()] = format_name
        
        logger.info(f"Registered format handler: {format_name}")
    
    def get_handler(self, format_name: str, **kwargs) -> BaseFormatHandler:
        """
        Get a format handler instance
        
        Args:
            format_name: Name of the format
            **kwargs: Arguments to pass to handler constructor
            
        Returns:
            Format handler instance
        """
        if format_name not in self._handlers:
            raise ValueError(f"Unknown format: {format_name}")
        
        handler_class = self._handlers[format_name]
        return handler_class(**kwargs)
    
    def get_handler_by_extension(self, file_path: str, **kwargs) -> Optional[BaseFormatHandler]:
        """
        Get a format handler by file extension
        
        Args:
            file_path: Path to the file
            **kwargs: Arguments to pass to handler constructor
            
        Returns:
            Format handler instance or None if not found
        """
        import os
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in self._extensions:
            format_name = self._extensions[ext]
            return self.get_handler(format_name, **kwargs)
        
        return None
    
    def get_handler_by_path(self, path: str, **kwargs) -> Optional[BaseFormatHandler]:
        """
        Get a format handler by trying to load the path
        
        Args:
            path: Path to the dataset
            **kwargs: Arguments to pass to handler constructor
            
        Returns:
            Format handler instance or None if not found
        """
        # First try direct path detection
        for format_name, handler_class in self._handlers.items():
            try:
                handler = handler_class(**kwargs)
                if handler.can_load(path):
                    return handler
            except Exception:
                continue
        
        # If path is a directory, try to find bin/idx files
        if os.path.isdir(path):
            for file in os.listdir(path):
                if file.endswith('.bin'):
                    # Found a .bin file, check if corresponding .idx exists
                    base_name = file[:-4]  # Remove .bin extension
                    idx_file = os.path.join(path, f"{base_name}.idx")
                    if os.path.exists(idx_file):
                        # Try to load with binidx handler
                        binidx_path = os.path.join(path, base_name)
                        try:
                            binidx_handler = self._handlers.get('binidx')
                            if binidx_handler:
                                handler = binidx_handler(**kwargs)
                                if handler.can_load(binidx_path):
                                    return handler
                        except Exception:
                            continue
        
        return None
    
    def list_formats(self) -> List[str]:
        """
        List all registered formats
        
        Returns:
            List of format names
        """
        return list(self._handlers.keys())
    
    def list_extensions(self) -> Dict[str, str]:
        """
        List all registered file extensions
        
        Returns:
            Dictionary mapping extensions to format names
        """
        return self._extensions.copy()
    
    def convert(self, input_path: str, output_path: str, 
                input_format: Optional[str] = None,
                output_format: Optional[str] = None,
                **kwargs) -> None:
        """
        Convert between formats
        
        Args:
            input_path: Path to input dataset
            output_path: Path to output dataset
            input_format: Input format name (auto-detect if None)
            output_format: Output format name (auto-detect if None)
            **kwargs: Additional conversion parameters
        """
        # Extract handler-specific parameters
        tokenizer_path = kwargs.pop('tokenizer_path', None)
        handler_kwargs = {'tokenizer_path': tokenizer_path}
        
        # Get input handler
        if input_format:
            input_handler = self.get_handler(input_format, **handler_kwargs)
        else:
            input_handler = self.get_handler_by_path(input_path, **handler_kwargs)
            if not input_handler:
                input_handler = self.get_handler_by_extension(input_path, **handler_kwargs)
        
        # For bin/idx format, we need to find the correct path
        actual_input_path = input_path
        if input_handler and input_handler.format_name == 'binidx' and os.path.isdir(input_path):
            # Find the bin/idx files in the directory
            for file in os.listdir(input_path):
                if file.endswith('.bin'):
                    base_name = file[:-4]  # Remove .bin extension
                    idx_file = os.path.join(input_path, f"{base_name}.idx")
                    if os.path.exists(idx_file):
                        actual_input_path = os.path.join(input_path, base_name)
                        break
        
        if not input_handler:
            raise ValueError(f"Cannot determine input format for: {input_path}")
        
        # Get output handler
        if output_format:
            output_handler = self.get_handler(output_format, **handler_kwargs)
        else:
            output_handler = self.get_handler_by_extension(output_path, **handler_kwargs)
            if not output_handler:
                raise ValueError(f"Cannot determine output format for: {output_path}")
        
        logger.info(f"Converting {input_handler.format_name} to {output_handler.format_name}")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")

        # Perform conversion
        # Load documents as Document objects
        documents = input_handler.load(actual_input_path)

        # Save documents
        output_handler.save(documents, output_path, **kwargs)

        logger.info("Conversion completed successfully")


# Global registry instance
registry = FormatRegistry()
