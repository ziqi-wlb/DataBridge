"""
Tests for format handlers and registry
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch

from databridge.formats import (
    FormatRegistry,
    BaseFormatHandler,
    JsonlFormatHandler,
    WebDatasetFormatHandler,
    BinIdxFormatHandler,
    EnergonFormatHandler,
    registry,
)


class TestFormatRegistry:
    """Test format registry functionality"""
    
    def test_registry_initialization(self):
        """Test registry initialization"""
        reg = FormatRegistry()
        assert reg is not None
        assert isinstance(reg._handlers, dict)
        assert isinstance(reg._extensions, dict)
    
    def test_register_handler(self):
        """Test handler registration"""
        reg = FormatRegistry()
        
        # Create a mock handler
        class MockHandler(BaseFormatHandler):
            @property
            def format_name(self):
                return "mock"
            
            @property
            def file_extensions(self):
                return [".mock"]
            
            def can_load(self, path):
                return True
            
            def load(self, path):
                yield {"id": 0, "text": "test"}
            
            def save(self, documents, output_path, **kwargs):
                pass
        
        reg.register(MockHandler)
        assert "mock" in reg._handlers
        assert ".mock" in reg._extensions
    
    def test_get_handler(self):
        """Test getting handler by name"""
        handler = registry.get_handler("jsonl")
        assert isinstance(handler, JsonlFormatHandler)
    
    def test_get_handler_by_extension(self):
        """Test getting handler by file extension"""
        handler = registry.get_handler_by_extension("test.jsonl")
        assert isinstance(handler, JsonlFormatHandler)
    
    def test_list_formats(self):
        """Test listing all formats"""
        formats = registry.list_formats()
        assert "jsonl" in formats
        assert "webdataset" in formats
        assert "binidx" in formats
        assert "energon" in formats
    
    def test_list_extensions(self):
        """Test listing all extensions"""
        extensions = registry.list_extensions()
        assert ".jsonl" in extensions
        assert ".tar" in extensions


class TestJsonlFormatHandler:
    """Test JSONL format handler"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_jsonl = os.path.join(self.temp_dir, "test.jsonl")
        
        # Create test JSONL file
        test_data = [
            {"id": 0, "text": "Hello world"},
            {"id": 1, "text": "Test document"},
            {"id": 2, "text": "Another sample"}
        ]
        
        with open(self.test_jsonl, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_format_name(self):
        """Test format name property"""
        handler = JsonlFormatHandler()
        assert handler.format_name == "jsonl"
    
    def test_file_extensions(self):
        """Test file extensions property"""
        handler = JsonlFormatHandler()
        assert ".jsonl" in handler.file_extensions
        assert ".json" in handler.file_extensions
    
    def test_can_load(self):
        """Test can_load method"""
        handler = JsonlFormatHandler()
        assert handler.can_load(self.test_jsonl) is True
        assert handler.can_load("/nonexistent/file") is False
    
    def test_load(self):
        """Test load method"""
        handler = JsonlFormatHandler()
        documents = list(handler.load(self.test_jsonl))
        
        assert len(documents) == 3
        assert documents[0]["id"] == 0
        assert documents[0]["text"] == "Hello world"
        assert documents[1]["id"] == 1
        assert documents[1]["text"] == "Test document"
    
    def test_save(self):
        """Test save method"""
        handler = JsonlFormatHandler()
        output_file = os.path.join(self.temp_dir, "output.jsonl")
        
        documents = [
            {"id": 0, "text": "Test 1"},
            {"id": 1, "text": "Test 2"}
        ]
        
        handler.save(iter(documents), output_file)
        
        # Verify output
        assert os.path.exists(output_file)
        with open(output_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert json.loads(lines[0])["text"] == "Test 1"
            assert json.loads(lines[1])["text"] == "Test 2"
    
    def test_get_document_count(self):
        """Test get_document_count method"""
        handler = JsonlFormatHandler()
        count = handler.get_document_count(self.test_jsonl)
        assert count == 3


class TestWebDatasetFormatHandler:
    """Test WebDataset format handler"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_webdataset_dir = os.path.join(self.temp_dir, "webdataset")
        os.makedirs(self.test_webdataset_dir, exist_ok=True)
        
        # Create mock WebDataset files
        with open(os.path.join(self.test_webdataset_dir, "shard_000000.tar"), 'wb') as f:
            f.write(b"mock_tar_data")
        
        # Create dataset info
        info = {
            "num_documents": 100,
            "num_shards": 1,
            "shard_size": 100
        }
        with open(os.path.join(self.test_webdataset_dir, "dataset_info.json"), 'w') as f:
            json.dump(info, f)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_format_name(self):
        """Test format name property"""
        handler = WebDatasetFormatHandler()
        assert handler.format_name == "webdataset"
    
    def test_can_load(self):
        """Test can_load method"""
        handler = WebDatasetFormatHandler()
        assert handler.can_load(self.test_webdataset_dir) is True
        assert handler.can_load("/nonexistent/dir") is False


class TestBaseFormatHandler:
    """Test base format handler functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_validate_path(self):
        """Test path validation"""
        # Create a test file
        test_file = os.path.join(self.temp_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test content")
        
        # Create a concrete handler for testing
        class TestHandler(BaseFormatHandler):
            @property
            def format_name(self):
                return "test"
            
            @property
            def file_extensions(self):
                return [".txt"]
            
            def can_load(self, path):
                return True
            
            def load(self, path):
                yield {"id": 0, "text": "test"}
            
            def save(self, documents, output_path, **kwargs):
                pass
        
        handler = TestHandler()
        
        # Test valid path
        assert handler.validate_path(test_file) is True
        
        # Test invalid path
        assert handler.validate_path("/nonexistent/file") is False
    
    def test_tokenizer_loading(self):
        """Test tokenizer loading"""
        with patch('databridge.formats.base.AutoTokenizer') as mock_tokenizer:
            mock_tokenizer.from_pretrained.return_value = Mock()
            
            class TestHandler(BaseFormatHandler):
                @property
                def format_name(self):
                    return "test"
                
                @property
                def file_extensions(self):
                    return [".txt"]
                
                def can_load(self, path):
                    return True
                
                def load(self, path):
                    yield {"id": 0, "text": "test"}
                
                def save(self, documents, output_path, **kwargs):
                    pass
            
            handler = TestHandler(tokenizer_path="/fake/path")
            assert handler.tokenizer is not None
            mock_tokenizer.from_pretrained.assert_called_once()


class TestConversion:
    """Test format conversion functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_jsonl = os.path.join(self.temp_dir, "input.jsonl")
        self.output_dir = os.path.join(self.temp_dir, "output")
        
        # Create test JSONL file
        test_data = [
            {"id": 0, "text": "Hello world"},
            {"id": 1, "text": "Test document"}
        ]
        
        with open(self.input_jsonl, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_jsonl_to_webdataset_conversion(self):
        """Test JSONL to WebDataset conversion"""
        # This would test the actual conversion
        # For now, we'll just test that the registry can handle it
        input_handler = registry.get_handler("jsonl")
        output_handler = registry.get_handler("webdataset")
        
        assert isinstance(input_handler, JsonlFormatHandler)
        assert isinstance(output_handler, WebDatasetFormatHandler)
        
        # Test that we can load from JSONL
        documents = list(input_handler.load(self.input_jsonl))
        assert len(documents) == 2
