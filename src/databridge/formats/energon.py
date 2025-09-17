"""
Megatron-Energon format handler
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


class EnergonFormatHandler(BaseFormatHandler):
    """Handler for Megatron-Energon format"""
    
    @property
    def format_name(self) -> str:
        return "energon"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".tar"]
    
    def can_load(self, path: str) -> bool:
        """Check if this handler can load the given path"""
        if not self.validate_path(path):
            return False
        
        # Check if it's a directory with Energon files
        if not os.path.isdir(path):
            return False
        
        # Look for shard files and dataset_info.json
        dataset_info_file = os.path.join(path, "dataset_info.json")
        if not os.path.exists(dataset_info_file):
            return False
        
        # Check for at least one shard file
        shard_files = [f for f in os.listdir(path) if f.startswith('shard_') and f.endswith('.tar')]
        return len(shard_files) > 0
    
    def load(self, path: str) -> Iterator[Document]:
        """Load documents from Energon format"""
        logger.info(f"Loading Energon dataset from {path}")
        
        # Load dataset info
        info_file = os.path.join(path, "dataset_info.json")
        if os.path.exists(info_file):
            with open(info_file, 'r') as f:
                dataset_info = json.load(f)
            logger.info(f"Dataset info: {dataset_info}")
        
        # Find all shard files
        shard_files = [f for f in os.listdir(path) if f.startswith('shard_') and f.endswith('.tar')]
        shard_files.sort()  # Sort to ensure consistent order
        logger.info(f"Found {len(shard_files)} shard files")
        
        # Iterate through shards
        for shard_file in tqdm(shard_files, desc="Loading shards"):
            full_shard_path = os.path.join(path, shard_file)
            dataset = wds.WebDataset(full_shard_path)
            
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
                
                doc_data = {
                    'id': metadata.get('id', 'unknown'),
                    'text': text,
                    'tokens': tokens,
                    'length': metadata.get('length', len(text)),
                    'token_count': metadata.get('token_count', len(tokens))
                }
                yield Document(doc_data)
        
        logger.info(f"Finished loading Energon dataset from {path}")
    
    def save(self, documents: Iterator[Document], output_path: str, **kwargs) -> None:
        """Save documents to Energon format using native energon prepare command"""
        shard_size = kwargs.get('shard_size', 1000)
        
        logger.info(f"Saving Energon dataset to {output_path}")
        logger.info(f"Shard size: {shard_size}")
        
        # Create temporary WebDataset format first
        temp_webdataset_path = f"{output_path}_temp_webdataset"
        logger.info(f"Creating temporary WebDataset at {temp_webdataset_path}")
        
        # Use WebDataset format handler to create the base format
        from .webdataset import WebDatasetFormatHandler
        webdataset_handler = WebDatasetFormatHandler()
        webdataset_handler.save(documents, temp_webdataset_path, shard_size=shard_size)
        
        # Now use energon prepare command to convert to proper Energon format
        logger.info("Converting WebDataset to Energon format using native energon prepare command")
        self._convert_webdataset_to_energon(temp_webdataset_path, output_path)
        
        # Move the prepared dataset to the final output path
        import shutil
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        shutil.move(temp_webdataset_path, output_path)
        
        logger.info(f"Finished saving Energon dataset to {output_path}")
    
    def _convert_webdataset_to_energon(self, webdataset_path: str, energon_path: str) -> None:
        """Convert WebDataset to Energon format using native energon prepare command"""
        import subprocess
        import glob
        
        # Find actual shard files to determine the correct split-parts range
        shard_files = glob.glob(os.path.join(webdataset_path, "shard_*.tar"))
        shard_files.sort()
        
        if not shard_files:
            raise RuntimeError(f"No shard files found in {webdataset_path}")
        
        # Extract shard numbers to determine the range
        shard_numbers = []
        for shard_file in shard_files:
            basename = os.path.basename(shard_file)
            # Extract number from shard_XXXXXX.tar
            shard_num = basename.replace('shard_', '').replace('.tar', '')
            shard_numbers.append(int(shard_num))
        
        min_shard = 0  # Always start from 0
        max_shard = max(shard_numbers)
        
        # Build split-parts pattern using wildcard to match all tar files
        split_parts = "train:shard_.*.tar"
        
        # Build energon prepare command with correct split-parts
        cmd = [
            "energon", "prepare", webdataset_path,
            "--progress",
            "--split-parts", split_parts,
            "--num-workers", "32",
            "--shuffle-tars"
        ]
        
        logger.info(f"Found {len(shard_files)} shard files (range: {min_shard:06d}-{max_shard:06d})")
        logger.info(f"Running command: {' '.join(cmd)}")

        # Run energon prepare command (it modifies the webdataset_path in-place)
        # Check if dataset already exists to determine if we need the first 'y'
        dataset_info_path = os.path.join(webdataset_path, ".nv-meta", ".info.json")
        dataset_exists = os.path.exists(dataset_info_path)
        
        if dataset_exists:
            # Dataset exists, need to respond to "continue?" prompt
            auto_input = "y\ny\n8\ny\ntext\n"  # continue, create yaml, TextSample, simple mapping, text field
        else:
            # New dataset, no "continue?" prompt
            auto_input = "y\n8\ny\ntext\n"  # create yaml, TextSample, simple mapping, text field
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            input=auto_input
        )
        
        if result.returncode != 0:
            logger.error(f"energon prepare failed with return code {result.returncode}")
            logger.error(f"stdout: {result.stdout}")
            logger.error(f"stderr: {result.stderr}")
            raise RuntimeError(f"energon prepare command failed: {result.stderr}")
        
        logger.info("energon prepare completed successfully")
        logger.info(f"stdout: {result.stdout}")
        
        # energon prepare automatically generates correct split.yaml with wildcard pattern
        
    def _create_tar_index(self, tar_path: str, idx_path: str) -> None:
        """Create .tar.idx file for a tar file"""
        import tarfile
        
        # Create index file
        with open(idx_path, 'w') as idx_file:
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        idx_file.write(f"{member.name}\t{member.offset}\t{member.size}\n")
    
    def _create_energon_metadata(self, output_dir: str, num_documents: int, num_shards: int, shard_size: int, shard_files: list) -> None:
        """Create Energon-specific metadata files"""
        # Create dataset info (matching standard format)
        info_file = os.path.join(output_dir, "dataset_info.json")
        dataset_info = {
            "num_documents": num_documents,
            "num_shards": num_shards,
            "shard_size": shard_size,
            "fields": ["text"],
            "source_format": "jsonl",
            "note": "text-only webdataset"
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        # Create .nv-meta directory and files
        nv_meta_dir = os.path.join(output_dir, ".nv-meta")
        os.makedirs(nv_meta_dir, exist_ok=True)
        
        # Create .info.json
        info_json = {
            "version": "1.0",
            "num_samples": num_documents,
            "num_parts": num_shards,
            "part_types": ["text"],
            "sample_type": "TextSample"
        }
        
        with open(os.path.join(nv_meta_dir, ".info.json"), 'w') as f:
            json.dump(info_json, f, indent=2)
        
        # Create dataset.yaml
        dataset_yaml = """sample_type:
  __module__: megatron.energon
  __class__: TextSample
sample_loader: sample_loader.py:sample_loader
part_filter: sample_loader.py:part_filter
"""
        
        with open(os.path.join(nv_meta_dir, "dataset.yaml"), 'w') as f:
            f.write(dataset_yaml)
        
        # Create split.yaml
        shard_list = [f'"{shard}"' for shard in shard_files]
        split_yaml = f"""split_parts:
  train: [{', '.join(shard_list)}]           
exclude: []
"""
        
        with open(os.path.join(nv_meta_dir, "split.yaml"), 'w') as f:
            f.write(split_yaml)
        
        # Create sample_loader.py
        sample_loader_py = '''def sample_loader(sample):
    """Load a sample from the dataset."""
    return sample

def part_filter(part):
    """Filter parts for the dataset."""
    return True
'''
        
        with open(os.path.join(nv_meta_dir, "sample_loader.py"), 'w') as f:
            f.write(sample_loader_py)
        
        # Create index.uuid
        import uuid
        with open(os.path.join(nv_meta_dir, "index.uuid"), 'w') as f:
            f.write(str(uuid.uuid4()))
        
        # Create empty index.sqlite (placeholder)
        import sqlite3
        conn = sqlite3.connect(os.path.join(nv_meta_dir, "index.sqlite"))
        conn.close()
        
        logger.info(f"Created Energon metadata files in {nv_meta_dir}")
