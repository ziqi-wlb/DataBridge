"""
Progress tracking utilities
"""

import time
import logging
from typing import Optional, Callable
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks progress of dataset conversion operations"""
    
    def __init__(self, total: int, desc: str = "Processing", unit: str = "items"):
        """
        Initialize progress tracker
        
        Args:
            total: Total number of items to process
            desc: Description for the progress bar
            unit: Unit of measurement
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self.pbar = None
        
    def __enter__(self):
        """Context manager entry"""
        self.pbar = tqdm(
            total=self.total,
            desc=self.desc,
            unit=self.unit,
            dynamic_ncols=True
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.pbar:
            self.pbar.close()
    
    def update(self, n: int = 1) -> None:
        """
        Update progress
        
        Args:
            n: Number of items processed
        """
        self.current += n
        if self.pbar:
            self.pbar.update(n)
    
    def set_description(self, desc: str) -> None:
        """
        Update progress bar description
        
        Args:
            desc: New description
        """
        if self.pbar:
            self.pbar.set_description(desc)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        return time.time() - self.start_time
    
    def get_eta(self) -> Optional[float]:
        """Get estimated time to completion in seconds"""
        if self.current == 0:
            return None
        
        elapsed = self.get_elapsed_time()
        rate = self.current / elapsed
        remaining = self.total - self.current
        
        if rate > 0:
            return remaining / rate
        return None
    
    def get_progress_percentage(self) -> float:
        """Get progress as percentage"""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    def log_progress(self, level: str = "INFO") -> None:
        """Log current progress"""
        elapsed = self.get_elapsed_time()
        eta = self.get_eta()
        percentage = self.get_progress_percentage()
        
        eta_str = f"ETA: {eta:.1f}s" if eta else "ETA: unknown"
        
        log_msg = f"Progress: {self.current}/{self.total} ({percentage:.1f}%) - Elapsed: {elapsed:.1f}s - {eta_str}"
        
        if level.upper() == "DEBUG":
            logger.debug(log_msg)
        elif level.upper() == "INFO":
            logger.info(log_msg)
        elif level.upper() == "WARNING":
            logger.warning(log_msg)
        elif level.upper() == "ERROR":
            logger.error(log_msg)


class ConversionProgressTracker(ProgressTracker):
    """Specialized progress tracker for dataset conversions"""
    
    def __init__(self, total_documents: int, conversion_type: str):
        """
        Initialize conversion progress tracker
        
        Args:
            total_documents: Total number of documents to convert
            conversion_type: Type of conversion being performed
        """
        super().__init__(total_documents, f"Converting to {conversion_type}", "docs")
        self.conversion_type = conversion_type
        self.successful_conversions = 0
        self.failed_conversions = 0
    
    def mark_success(self) -> None:
        """Mark a successful conversion"""
        self.successful_conversions += 1
        self.update(1)
    
    def mark_failure(self) -> None:
        """Mark a failed conversion"""
        self.failed_conversions += 1
        self.update(1)
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage"""
        if self.current == 0:
            return 0.0
        return (self.successful_conversions / self.current) * 100
    
    def log_summary(self) -> None:
        """Log conversion summary"""
        success_rate = self.get_success_rate()
        elapsed = self.get_elapsed_time()
        
        logger.info(f"Conversion completed:")
        logger.info(f"  Total documents: {self.total}")
        logger.info(f"  Successful: {self.successful_conversions}")
        logger.info(f"  Failed: {self.failed_conversions}")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        logger.info(f"  Total time: {elapsed:.1f}s")
        logger.info(f"  Average speed: {self.current/elapsed:.1f} docs/s") 