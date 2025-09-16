"""
DataBridge - A comprehensive dataset conversion toolkit
"""

__version__ = "0.1.0"
__author__ = "ZiqiYu"
__email__ = "550461236@qq.com"

from .comm.dataset import Dataset, Document, DatasetCollection
from .comm.dataloader import (
    DataLoader,
    PyTorchDataLoader,
    HuggingFaceDataLoader,
    MegatronDataLoader,
    DataLoaderFactory,
    create_pytorch_loader,
    create_huggingface_loader,
    create_megatron_loader,
)
from .formats import (
    FormatRegistry,
    BaseFormatHandler,
    JsonlFormatHandler,
    WebDatasetFormatHandler,
    BinIdxFormatHandler,
    EnergonFormatHandler,
)
from .formats.registry import registry

# Register default format handlers
registry.register(JsonlFormatHandler)
registry.register(WebDatasetFormatHandler)
registry.register(BinIdxFormatHandler)
registry.register(EnergonFormatHandler)

__all__ = [
    "Dataset",
    "Document", 
    "DatasetCollection",
    "DataLoader",
    "PyTorchDataLoader",
    "HuggingFaceDataLoader",
    "MegatronDataLoader",
    "DataLoaderFactory",
    "create_pytorch_loader",
    "create_huggingface_loader",
    "create_megatron_loader",
    "FormatRegistry",
    "BaseFormatHandler", 
    "JsonlFormatHandler",
    "WebDatasetFormatHandler",
    "BinIdxFormatHandler",
    "EnergonFormatHandler",
    "registry",
]