# DataBridge

A comprehensive dataset conversion toolkit for transforming between different dataset formats commonly used in machine learning and NLP tasks.

## 🚀 Features

DataBridge is a comprehensive dataset conversion toolkit that supports seamless transformation between different dataset formats commonly used in machine learning and NLP tasks.

### Supported Formats

- **JSONL** - JSON Lines format for text data
- **Megatron bin/idx** - Binary format used by Megatron-LM
- **WebDataset** - Tar-based dataset format for large-scale training
- **Energon Dataset** - Megatron-Energon compatible format (fully compatible with VeOmni training framework)

### Key Capabilities

- **Universal Conversion**: Convert between any supported format pair
- **VeOmni Ready**: Native support for Energon format used by VeOmni training framework
- **Progress Tracking**: Real-time progress bars and detailed logging
- **Runtime Loading**: Support for both offline conversion and online data loading


## 📦 Installation

### From Source

```bash
cd DataBridge
pip install -e .
```

### Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Command Line Interface

DataBridge provides a unified command-line interface for all format conversions:

```bash
# List all supported formats
databridge list-formats

# Convert between any supported formats
databridge convert \
    --input-path /path/to/input \
    --output-path /path/to/output \
    --input-format <input_format> \
    --output-format <output_format> \
    --shard-size 1000
```

#### Common Conversion Examples

**1. Convert Megatron bin/idx to Energon format (for VeOmni training):**
```bash
# Convert bin/idx dataset to Energon format
databridge convert \
    --input-path /path/to/dataset \
    --output-path /path/to/output/energon_dataset \
    --input-format binidx \
    --output-format energon \
    --shard-size 1000

# Example with real paths:
databridge convert \
    --input-path /prodcpfs/user/weishi/data/text_data/pile_test \
    --output-path /prodcpfs/user/weishi/data/text_data_converted/pile_test/energon \
    --input-format binidx \
    --output-format energon \
    --shard-size 1000
```

**2. Convert JSONL to WebDataset:**
```bash
databridge convert \
    --input-path data.jsonl \
    --output-path webdataset/ \
    --input-format jsonl \
    --output-format webdataset \
    --shard-size 1000
```

**3. Convert bin/idx to JSONL:**
```bash
databridge convert \
    --input-path dataset \
    --output-path data.jsonl \
    --input-format binidx \
    --output-format jsonl
```

### Python API

#### Using the Registry (Recommended)

```python
from databridge.formats.registry import registry

# Convert bin/idx to Energon format
registry.convert(
    input_path="/path/to/binidx/dataset",
    output_path="/path/to/output/energon",
    input_format="binidx",
    output_format="energon",
    shard_size=1000
)

# Convert JSONL to WebDataset
registry.convert(
    input_path="data.jsonl",
    output_path="webdataset/",
    input_format="jsonl",
    output_format="webdataset",
    shard_size=1000
)

# List available formats
formats = registry.list_formats()
print(f"Supported formats: {formats}")
```

#### Using Individual Format Handlers

```python
from databridge import Document, JsonlFormatHandler, WebDatasetFormatHandler, BinIdxFormatHandler, EnergonFormatHandler

# Load data using specific handlers
jsonl_handler = JsonlFormatHandler()
binidx_handler = BinIdxFormatHandler()
webdataset_handler = WebDatasetFormatHandler()
energon_handler = EnergonFormatHandler()

# Load returns Document objects
documents = jsonl_handler.load("data.jsonl")
# or
documents = binidx_handler.load("/path/to/binidx/dataset")

# Save accepts Document objects
webdataset_handler.save(documents, "webdataset/", shard_size=1000)
# or
energon_handler.save(documents, "energon/", shard_size=1000)

# Work with Document objects directly
for doc in documents:
    print(f"Document {doc.doc_id}: {doc['text']}")
    # Access data like a dictionary
    if 'metadata' in doc:
        print(f"Metadata: {doc['metadata']}")
```

## 🎯 Common Use Cases

### Converting Megatron bin/idx to Energon for VeOmni Training

This is the most common use case for DataBridge - converting existing Megatron bin/idx datasets to Energon format for use with VeOmni training framework.

#### Step 1: Prepare Your Data

Ensure your bin/idx dataset has the following structure:
```
/path/to/your/dataset/
├── dataset.bin          # Binary data file
└── dataset.idx          # Index file
```

#### Step 2: Convert to Energon Format

```bash
# Convert bin/idx to Energon format
databridge convert \
    --input-path /path/to/your/dataset \
    --output-path /path/to/output/energon_dataset \
    --input-format binidx \
    --output-format energon \
    --shard-size 1000
```

#### Step 3: Verify the Output

The output Energon dataset will have the following structure:
```
/path/to/output/energon_dataset/
├── .nv-meta/
│   ├── .info.json       # Dataset metadata
│   ├── dataset.yaml     # Dataset configuration
│   ├── split.yaml       # Split configuration
│   ├── index.sqlite     # Index database
│   └── index.uuid       # Unique identifier
├── shard_000000.tar     # Data shards
├── shard_000000.tar.idx # Shard indices
├── shard_000001.tar
├── shard_000001.tar.idx
└── ...
```

#### Step 4: Use with VeOmni

Update your VeOmni training script to use the Energon dataset:

```bash
# In your VeOmni debug.sh or training script
DATA_PATH=/path/to/output/energon_dataset
DATA_SET_TYPE=energon

# Run training
bash train.sh tasks/train_torch.py $CONFIG \
    --data.train_path $DATA_PATH \
    --data.datasets_type $DATA_SET_TYPE \
    --train.global_batch_size 128 \
    --train.lr 5e-7
```


## Runtime Data Loading(WIP)

DataBridge also supports runtime dataset loading for training frameworks:

### PyTorch Integration

```python
from databridge import create_pytorch_loader

# Create PyTorch data loader
loader = create_pytorch_loader(
    dataset_path="data.jsonl",
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Use in training loop
for batch in loader:
    texts = batch['text']  # List of texts
    ids = batch['id']      # Tensor of IDs
    # Process batch...
```

### HuggingFace Integration

```python
from databridge import create_huggingface_loader

# Create HuggingFace loader
loader = create_huggingface_loader(dataset_path="data.jsonl")

# Convert to HuggingFace Dataset
hf_dataset = loader.to_huggingface_dataset()

# Use in training
for doc in loader:
    text = doc['text']
    doc_id = doc['id']
    # Process document...
```

### Megatron Integration

```python
from databridge import create_megatron_loader

# Create Megatron loader with tokenizer
loader = create_megatron_loader(
    dataset_path="data.jsonl",
    tokenizer_path="/path/to/tokenizer"
)

# Get tokenized data
for doc in loader:
    tokens = doc['tokens']
    # Process tokenized document...
```


### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/databridge

# Run specific test file
pytest tests/test_converters.py
```

