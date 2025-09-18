"""
Command line interface for DataBridge
"""

import os
import sys
import logging
import click
from rich.console import Console
from rich.logging import RichHandler

from .formats.registry import registry
from .comm.dataloader import DataLoaderFactory
from . import __version__

# Setup rich console
console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)

logger = logging.getLogger("data-bridge")


def setup_logging(verbose: bool, quiet: bool):
    """Setup logging level based on verbosity flags"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)
    else:
        logging.getLogger().setLevel(logging.INFO)


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress logging output")
def main(verbose: bool, quiet: bool):
    """DataBridge - Dataset format conversion toolkit"""
    setup_logging(verbose, quiet)


@main.command()
@click.option("--input", "-i", required=True, help="Input dataset path")
@click.option("--output", "-o", required=True, help="Output dataset path")
@click.option("--input-format", help="Input format (auto-detect if not specified)")
@click.option("--output-format", help="Output format (auto-detect if not specified)")
@click.option("--tokenizer-path", "-t", help="Path to tokenizer")
@click.option("--shard-size", "-s", default=1000, type=int, help="Documents per shard")
@click.option("--force", "-f", is_flag=True, help="Overwrite existing output")
def convert(input: str, output: str, input_format: str, output_format: str, 
            tokenizer_path: str, shard_size: int, force: bool):
    """Convert between dataset formats"""
    try:
        # Validate input - special handling for bin/idx format
        input_exists = False
        if input_format == 'binidx':
            # For bin/idx format, check if .bin and .idx files exist
            bin_file = f"{input}.bin"
            idx_file = f"{input}.idx"
            input_exists = os.path.exists(bin_file) and os.path.exists(idx_file)
        else:
            input_exists = os.path.exists(input)
        
        if not input_exists:
            console.print(f"[red]Error: Input path does not exist: {input}[/red]")
            if input_format == 'binidx':
                console.print(f"[red]Expected files: {input}.bin and {input}.idx[/red]")
            sys.exit(1)
        
        # Check output
        if os.path.exists(output) and not force:
            console.print(f"[yellow]Warning: Output path exists: {output}[/yellow]")
            if not click.confirm("Do you want to overwrite it?"):
                sys.exit(0)
        
        # Run conversion
        console.print(f"[blue]Converting dataset...[/blue]")
        registry.convert(
            input_path=input,
            output_path=output,
            input_format=input_format,
            output_format=output_format,
            tokenizer_path=tokenizer_path,
            shard_size=shard_size
        )
        
        console.print(f"[green]Conversion completed successfully![/green]")
        console.print(f"Output: {output}")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
def list_formats():
    """List all supported formats"""
    formats = registry.list_formats()
    extensions = registry.list_extensions()
    
    console.print("[bold blue]Supported Formats:[/bold blue]")
    for format_name in formats:
        console.print(f"  • {format_name}")
    
    console.print("\n[bold blue]File Extensions:[/bold blue]")
    for ext, format_name in extensions.items():
        console.print(f"  • {ext} → {format_name}")


@main.command()
def info():
    """Show DataBridge information"""
    console.print("[bold blue]DataBridge - Dataset Format Conversion Toolkit[/bold blue]")
    console.print(f"Version: {__version__}")
    console.print(f"Supported formats:")
    
    formats = registry.list_formats()
    for format_name in formats:
        console.print(f"  • {format_name}")
    
    console.print(f"\nFor more information, visit: https://github.com/ziqi-wlb/DataBridge")


@main.command()
@click.option('--input', '-i', required=True, help='Input dataset path')
@click.option('--loader-type', '-t', required=True, 
              type=click.Choice(['pytorch', 'huggingface', 'megatron']),
              help='Type of data loader to create')
@click.option('--format', '-f', help='Input format (auto-detect if not specified)')
@click.option('--tokenizer', help='Path to tokenizer')
@click.option('--batch-size', default=1, help='Batch size for PyTorch loader')
@click.option('--shuffle', is_flag=True, help='Shuffle data for PyTorch loader')
@click.option('--num-workers', default=0, help='Number of workers for PyTorch loader')
@click.option('--samples', default=5, help='Number of samples to show')
def load(input, loader_type, format, tokenizer, batch_size, shuffle, num_workers, samples):
    """Load dataset using specified data loader"""
    try:
        # Create data loader
        loader_kwargs = {}
        if loader_type == 'pytorch':
            loader_kwargs.update({
                'batch_size': batch_size,
                'shuffle': shuffle,
                'num_workers': num_workers
            })
        
        loader = DataLoaderFactory.create_loader(
            loader_type=loader_type,
            dataset_path=input,
            format_name=format,
            tokenizer_path=tokenizer,
            **loader_kwargs
        )
        
        console.print(f"✅ Created {loader_type} data loader")
        console.print(f"📊 Dataset size: {len(loader)} items")
        console.print(f"📁 Format: {loader.handler.format_name}")
        
        # Show sample data
        console.print(f"\n🔍 Showing first {samples} samples:")
        
        count = 0
        for item in loader:
            if count >= samples:
                break
            
            console.print(f"\n--- Sample {count + 1} ---")
            if loader_type == 'pytorch':
                # PyTorch loader returns batches
                console.print(f"Batch keys: {list(item.keys())}")
                if 'text' in item:
                    console.print(f"Texts: {item['text'][:2]}...")  # Show first 2 texts
                if 'id' in item:
                    console.print(f"IDs: {item['id'].tolist()[:5]}...")  # Show first 5 IDs
            else:
                # Other loaders return individual items
                console.print(f"Keys: {list(item.keys())}")
                if 'text' in item:
                    text = item['text'][:100] + "..." if len(item['text']) > 100 else item['text']
                    console.print(f"Text: {text}")
                if 'id' in item:
                    console.print(f"ID: {item['id']}")
            
            count += 1
        
        console.print(f"\n✅ Successfully loaded {count} samples")
        
    except Exception as e:
        console.print(f"❌ Error loading dataset: {e}")
        sys.exit(1)


@main.command()
def list_loaders():
    """List available data loader types"""
    from rich.table import Table
    
    table = Table(title="Available Data Loaders")
    table.add_column("Type", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Framework", style="magenta")
    
    loaders = [
        ("pytorch", "PyTorch-compatible data loader with batching", "PyTorch"),
        ("huggingface", "HuggingFace datasets-compatible loader", "HuggingFace"),
        ("megatron", "Megatron-compatible tokenized data loader", "Megatron")
    ]
    
    for loader_type, description, framework in loaders:
        table.add_row(loader_type, description, framework)
    
    console.print(table)


if __name__ == "__main__":
    main() 