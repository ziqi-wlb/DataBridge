"""
Tokenizer management utilities
"""

import os
import logging
from typing import Optional, Union, List
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenizerManager:
    """Manages tokenizer loading and caching"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize tokenizer manager
        
        Args:
            cache_dir: Directory to cache tokenizers
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/databridge/tokenizers")
        self._tokenizers = {}
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_tokenizer(self, tokenizer_path: str, trust_remote_code: bool = True) -> Optional[AutoTokenizer]:
        """
        Get or load a tokenizer
        
        Args:
            tokenizer_path: Path to the tokenizer
            trust_remote_code: Whether to trust remote code
            
        Returns:
            Loaded tokenizer or None if failed
        """
        if tokenizer_path in self._tokenizers:
            return self._tokenizers[tokenizer_path]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=trust_remote_code,
                cache_dir=self.cache_dir
            )
            self._tokenizers[tokenizer_path] = tokenizer
            logger.info(f"Loaded tokenizer from {tokenizer_path}")
            return tokenizer
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            return None
    
    def encode_text(self, tokenizer_path: str, text: str, **kwargs) -> Optional[List[int]]:
        """
        Encode text using specified tokenizer
        
        Args:
            tokenizer_path: Path to the tokenizer
            text: Text to encode
            
        Returns:
            List of token IDs or None if failed
        """
        tokenizer = self.get_tokenizer(tokenizer_path)
        if tokenizer is None:
            return None
        
        try:
            tokens = tokenizer.encode(text, **kwargs)
            return tokens
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            return None
    
    def decode_tokens(self, tokenizer_path: str, token_ids: List[int], **kwargs) -> Optional[str]:
        """
        Decode token IDs to text using specified tokenizer
        
        Args:
            tokenizer_path: Path to the tokenizer
            token_ids: List of token IDs
            
        Returns:
            Decoded text or None if failed
        """
        tokenizer = self.get_tokenizer(tokenizer_path)
        if tokenizer is None:
            return None
        
        try:
            text = tokenizer.decode(token_ids, **kwargs)
            return text
        except Exception as e:
            logger.error(f"Failed to decode tokens: {e}")
            return None
    
    def get_vocab_size(self, tokenizer_path: str) -> Optional[int]:
        """
        Get vocabulary size of specified tokenizer
        
        Args:
            tokenizer_path: Path to the tokenizer
            
        Returns:
            Vocabulary size or None if failed
        """
        tokenizer = self.get_tokenizer(tokenizer_path)
        if tokenizer is None:
            return None
        
        try:
            return tokenizer.vocab_size
        except Exception as e:
            logger.error(f"Failed to get vocab size: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear loaded tokenizers cache"""
        self._tokenizers.clear()
        logger.info("Cleared tokenizer cache") 