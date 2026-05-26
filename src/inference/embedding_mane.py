"""
Phase 4: Embedding inference for MANE canonical transcripts with adaptive batching.
Uses NTv3 model via JAX/Flax with GPU acceleration.
Direct embedding of cDNA sequences (no chunking).
Configuration-driven using config_mane.yaml
"""

import logging
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import pyfaidx
import jax
import jax.numpy as jnp
from nucleotide_transformer_v3.pretrained import get_pretrained_ntv3_model
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure JAX to use GPU
jax.config.update('jax_platform_name', 'gpu')


class AdaptiveBatcher:
    """Group genes into batches based on token count to optimize GPU memory usage."""
    
    def __init__(self, max_tokens_per_batch: int = 120_000, pad_multiple: int = 1):
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pad_multiple = max(1, pad_multiple)

    def _round_up_to_pad_multiple(self, sequence_length: int) -> int:
        """Round sequence length up to the model padding multiple."""
        return ((sequence_length + self.pad_multiple - 1) // self.pad_multiple) * self.pad_multiple
    
    def create_batches(
        self,
        metadata: pd.DataFrame,
        max_sequence_length_nt: int,
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Create batches and filter oversized sequences before inference.
        
        Args:
            metadata: DataFrame with columns including 'sequence_length_nt'
        
        Returns:
            Tuple of (batches, skipped_indices)
        """
        valid_metadata = metadata[metadata['sequence_length_nt'] <= max_sequence_length_nt]
        skipped_indices = metadata[metadata['sequence_length_nt'] > max_sequence_length_nt].index.tolist()
        sorted_idx = valid_metadata.sort_values(by='sequence_length_nt').index.tolist()
        
        batches = []
        current_batch = []
        for idx in sorted_idx:
            longest_sequence_length = int(metadata.loc[idx, 'sequence_length_nt'])
            padded_sequence_length = self._round_up_to_pad_multiple(longest_sequence_length)
            projected_batch_size = len(current_batch) + 1
            projected_tokens = projected_batch_size * padded_sequence_length

            if projected_tokens <= self.max_tokens_per_batch:
                current_batch.append(idx)
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
        
        if current_batch:
            batches.append(current_batch)
        
        return batches, skipped_indices


class NTv3MANEEmbedder:
    """Generate embeddings for MANE cDNA sequences using NTv3 model with JAX/Flax."""
    
    def __init__(self, fasta_path: str, model_name: str = "NTv3_650M_pre", use_bfloat16: bool = True):
        """
        Args:
            fasta_path: Path to MANE cDNA sequences FASTA file (mane_sequences.fasta)
            model_name: Model identifier (e.g., "NTv3_650M_pre")
            use_bfloat16: Use bfloat16 for lower memory usage
        """
        logger.info(f"Loading model: {model_name} (bfloat16={use_bfloat16})")
        
        # Verify GPU configuration
        logger.info("\n" + "="*60)
        logger.info("GPU CONFIGURATION CHECK")
        logger.info("="*60)
        
        # Check JAX devices
        devices = jax.devices()
        gpu_devices = [d for d in devices if 'gpu' in d.platform.lower()]
        
        logger.info(f"JAX Platform Name: {jax.devices()[0].platform}")
        logger.info(f"Total JAX devices: {len(devices)}")
        logger.info(f"Available devices: {devices}")
        logger.info(f"GPU devices found: {len(gpu_devices)}")
        
        if len(gpu_devices) > 0:
            logger.info(f"✓ GPU is ENABLED and available")
            for i, dev in enumerate(gpu_devices):
                logger.info(f"  GPU {i}: {dev}")
        else:
            logger.warning("✗ GPU is NOT available - falling back to CPU")
        
        logger.info("="*60 + "\n")
        
        # Load model, tokenizer, and config
        self.pretrained_model, self.tokenizer, self.config = get_pretrained_ntv3_model(
            model_name=model_name,
            embeddings_layers_to_save=(6,),  # Save layer 6
            attention_maps_to_save=((6, 1),),
            use_bfloat16=use_bfloat16,
        )
        
        self.fasta = pyfaidx.Fasta(fasta_path)
        logger.info(f"Loaded FASTA file: {fasta_path}")
        logger.info(f"Number of sequences in FASTA: {len(self.fasta.keys())}")
        
        # Test GPU with a small computation
        logger.info("\nTesting GPU with small computation...")
        try:
            test_arr = jnp.ones((100, 100))
            result = jnp.dot(test_arr, test_arr)
            logger.info(f"✓ GPU test computation successful")
            logger.info(f"  Result shape: {result.shape}, device: {result.devices()}")
        except Exception as e:
            logger.warning(f"✗ GPU test failed: {e}")
    
    def get_sequence(self, sequence_id: str) -> str:
        """Load sequence from FASTA by sequence ID (transcript ID)."""
        fasta_key = None
        # Try exact match first
        if sequence_id in self.fasta:
            fasta_key = sequence_id
        else:
            # Try substring match (in case FASTA headers have prefixes)
            for key in self.fasta.keys():
                if sequence_id in key or key in sequence_id:
                    fasta_key = key
                    break
        
        if fasta_key is None:
            raise ValueError(f"Sequence ID {sequence_id} not found in FASTA. Available keys: {list(self.fasta.keys())[:5]}...")
        
        return str(self.fasta[fasta_key][:].seq).upper()
    
    def embed_batch(self, sequence_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for a batch of MANE cDNA sequences.
        
        Args:
            sequence_ids: List of sequence IDs (transcript IDs)
        
        Returns:
            (embeddings: np.ndarray [N, embedding_dim], valid_ids: List[str])
        """
        sequences = []
        for seq_id in sequence_ids:
            seq = self.get_sequence(seq_id)
            logger.info(
                f"Processing sequence {seq_id} with raw length {len(seq):,} bases"
            )
            sequences.append(seq)
        
        if not sequences:
            return np.array([]), []
        
        # Tokenize batch
        tokens = self.tokenizer.batch_np_tokenize(sequences)
        
        # Pad to multiple of 128 (2^num_downsamples)
        # Required by NTv3 model architecture
        num_downsamples = self.config.num_downsamples
        pad_multiple = 2 ** num_downsamples
        
        _, seq_length = tokens.shape
        padded_length = ((seq_length + pad_multiple - 1) // pad_multiple) * pad_multiple
        
        if padded_length > seq_length:
            padding = padded_length - seq_length
            tokens = np.pad(
                tokens,
                ((0, 0), (0, padding)),
                mode='constant',
                constant_values=self.tokenizer.pad_token_id
            )
            logger.debug(f"Padded tokens from {seq_length} to {padded_length} (multiple of {pad_multiple})")

        logger.info(
            f"Running model on token batch shape {tokens.shape} "
            f"(unpadded_length={seq_length:,}, padded_length={padded_length:,})"
        )

        # Run inference
        outs = self.pretrained_model(tokens)
        
        # Extract final embeddings (after deconv tower, restored to original resolution)
        final_embeddings = outs["embedding"]  # Shape: (B, L, embedding_dim) at token resolution
        
        # Compute mean embeddings (excluding padding)
        padding_mask = jnp.expand_dims(tokens != self.tokenizer.pad_token_id, axis=-1)
        masked_embeddings = final_embeddings * padding_mask
        
        # Mean pooling over all tokens to get one vector per sequence
        sequences_lengths = jnp.sum(padding_mask, axis=1)
        mean_embeddings = jnp.sum(masked_embeddings, axis=1) / sequences_lengths
        
        return np.array(mean_embeddings), sequence_ids


class EmbeddingPipeline:
    """Full pipeline: load FASTA, create adaptive batches, generate MANE embeddings."""
    
    def __init__(
        self,
        fasta_path: str,
        metadata_path: str,
        output_dir: str,
        embeddings_path: str,
        ids_path: str,
        model_name: str = "NTv3_650M_pre",
        max_tokens_per_batch: int = 120_000,
        max_sequence_length_nt: int = 120_000,
        use_bfloat16: bool = True,
        debug: bool = False
    ):
        self.fasta_path = fasta_path
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        self.embeddings_path = Path(embeddings_path)
        self.ids_path = Path(ids_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata (must include 'sequence_length_nt' column for adaptive batching)
        self.metadata = pd.read_csv(metadata_path, sep='\t')
        logger.info(f"Loaded metadata for {len(self.metadata)} genes")
        logger.info(f"Metadata columns: {list(self.metadata.columns)}")
        
        if 'sequence_length_nt' not in self.metadata.columns:
            raise ValueError("Metadata must include 'sequence_length_nt' column for adaptive batching")
        
        self.embedder = NTv3MANEEmbedder(
            fasta_path,
            model_name=model_name,
            use_bfloat16=use_bfloat16,
        )
        self.batcher = AdaptiveBatcher(
            max_tokens_per_batch=max_tokens_per_batch,
            pad_multiple=2 ** self.embedder.config.num_downsamples,
        )
        self.debug = debug
        self.max_sequence_length_nt = max_sequence_length_nt
        
        logger.info(f"Adaptive batcher configured with max_tokens_per_batch={max_tokens_per_batch:,}")
        logger.info(f"Max sequence length: {max_sequence_length_nt:,} nt")
    
    def debug_first_gene(self):
        """Show transformation of first gene through pipeline."""
        if not self.debug or len(self.metadata) == 0:
            return
        
        logger.info("\n" + "="*70)
        logger.info("DEBUG: First Gene Transformation (MANE cDNA Sequence)")
        logger.info("="*70)
        
        # Get first row from metadata
        first_row = self.metadata.iloc[0]
        
        # Get sequence ID (try sequence_id first, then transcript_id)
        if 'sequence_id' in self.metadata.columns:
            first_seq_id = first_row['sequence_id']
        elif 'transcript_id' in self.metadata.columns:
            first_seq_id = first_row['transcript_id']
        else:
            first_seq_id = first_row['gene_id']
        
        logger.info(f"\n1. GENE INFO:")
        logger.info(f"   Gene Symbol: {first_row.get('gene_name', 'N/A')}")
        logger.info(f"   Gene ID: {first_row.get('gene_id', 'N/A')}")
        logger.info(f"   Transcript ID: {first_row.get('transcript_id', 'N/A')}")
        logger.info(f"   Sequence Length (nt): {first_row.get('sequence_length_nt', 'N/A')}")
        
        # Load sequence
        logger.info(f"\n2. CDNA SEQUENCE:")
        seq = self.embedder.get_sequence(first_seq_id)
        logger.info(f"   Full length: {len(seq)} nt")
        logger.info(f"   First 100 bp: {seq[:100]}")
        if len(seq) > 100:
            logger.info(f"   Last 100 bp: {seq[-100:]}")

        if len(seq) > self.max_sequence_length_nt:
            logger.info(
                f"   Embedding skipped in debug: sequence length exceeds max "
                f"{self.max_sequence_length_nt:,} nt"
            )
            logger.info("="*70 + "\n")
            return
        
        # Tokenize
        logger.info(f"\n3. TOKENIZATION:")
        tokens = self.embedder.tokenizer.batch_np_tokenize([seq])
        token_ids = tokens[0].tolist()
        logger.info(f"   Sequence length: {len(seq)} nt")
        logger.info(f"   Token length (padded): {len(token_ids)}")
        logger.info(f"   First 20 tokens: {token_ids[:20]}")
        logger.info(f"   Token mapping: A=6, T=7, C=8, G=9, N=10, Pad=1")
        
        # Generate embedding
        logger.info(f"\n4. EMBEDDING GENERATION (NO CHUNKING):")
        embeddings, _ = self.embedder.embed_batch([first_seq_id])
        if len(embeddings) > 0:
            embedding = embeddings[0]
            embedding_stats = embedding.astype(np.float32)
            logger.info(f"   Embedding shape: {embedding.shape}")
            logger.info(f"   Embedding dtype: {embedding.dtype}")
            logger.info(f"   Mean value: {embedding_stats.mean():.6f}")
            logger.info(f"   Std value: {embedding_stats.std():.6f}")
            logger.info(f"   Min value: {embedding_stats.min():.6f}")
            logger.info(f"   Max value: {embedding_stats.max():.6f}")
            logger.info(f"   First 10 dimensions: {embedding[:10]}")
        
        logger.info("="*70 + "\n")

    def run(self) -> Dict[str, Any]:
        """
        Generate embeddings for all MANE cDNA sequences.

        Returns:
            Statistics dictionary
        """
        # Debug first gene
        self.debug_first_gene()
        
        # Create adaptive batches based on sequence lengths
        batches, skipped_indices = self.batcher.create_batches(
            self.metadata,
            max_sequence_length_nt=self.max_sequence_length_nt,
        )
        logger.info(f"Created {len(batches)} adaptive batches (max_tokens_per_batch={self.batcher.max_tokens_per_batch:,})")
        logger.info(
            f"Filtered {len(skipped_indices)} sequences longer than "
            f"{self.max_sequence_length_nt:,} nt before inference"
        )
        
        # Prepare output files
        embedding_file = self.embeddings_path
        gene_ids_file = self.ids_path
        stats_file = self.output_dir / "Embedding_stats_mane.txt"
        skipped_file = self.output_dir / "skipped_sequences_mane.txt"
        
        # Ensure output directories exist
        embedding_file.parent.mkdir(parents=True, exist_ok=True)
        gene_ids_file.parent.mkdir(parents=True, exist_ok=True)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        
        all_embeddings = []
        all_sequence_ids = []
        stats = {'total': len(self.metadata), 'success': 0, 'error': 0, 'skipped': len(skipped_indices)}
        
        # Process batches with progress bar
        for batch_idx, batch_indices in enumerate(tqdm(batches, desc="Processing batches", unit="batch")):
            batch_size = len(batch_indices)
            
            try:
                # Get sequence IDs from metadata rows
                if 'sequence_id' in self.metadata.columns:
                    sequence_ids = self.metadata.loc[batch_indices, 'sequence_id'].tolist()
                elif 'transcript_id' in self.metadata.columns:
                    sequence_ids = self.metadata.loc[batch_indices, 'transcript_id'].tolist()
                else:
                    sequence_ids = self.metadata.loc[batch_indices, 'gene_id'].tolist()
                
                # Get sequence lengths for logging
                sequence_lengths = self.metadata.loc[batch_indices, 'sequence_length_nt'].sum()
                
                embeddings, ids = self.embedder.embed_batch(sequence_ids)
                
                if len(embeddings) > 0:
                    all_embeddings.append(embeddings)
                    all_sequence_ids.extend(ids)
                    stats['success'] += len(ids)
                    logger.debug(f"Batch {batch_idx}: processed {len(ids)} sequences ({sequence_lengths:,} nt)")
                else:
                    logger.warning(f"Batch {batch_idx}: no embeddings generated")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {str(e)}")
                stats['error'] += batch_size
        
        # Concatenate all embeddings
        if all_embeddings:
            embeddings_matrix = np.vstack(all_embeddings)
            logger.info(f"Embedding matrix shape: {embeddings_matrix.shape}")
            
            # Save embeddings
            np.save(embedding_file, embeddings_matrix)
            logger.info(f"Saved embeddings to {embedding_file}")
            
            # Save sequence/gene IDs
            with open(gene_ids_file, 'w') as f:
                f.write('\n'.join(all_sequence_ids))
            logger.info(f"Saved gene IDs to {gene_ids_file}")
        
        # Save stats
        with open(stats_file, 'w') as f:
            f.write("Embedding Generation Summary (MANE cDNA Sequences - No Chunking)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Input FASTA: {self.fasta_path}\n")
            f.write(f"Metadata: {self.metadata_path}\n")
            f.write(f"Sequence Type: MANE canonical transcripts (cDNA)\n")
            f.write(f"Model: (check model_name from config_mane.yaml)\n")
            f.write(f"Framework: JAX/Flax\n")
            f.write(f"Chunking: DISABLED (full cDNA sequence per gene)\n")
            f.write(f"Pooling: Mean pooling over all tokens\n")
            f.write(f"Batching: Adaptive (max tokens per batch={self.batcher.max_tokens_per_batch:,})\n")
            f.write(f"Max sequence length: {self.max_sequence_length_nt:,} nt\n")
            f.write(f"Total genes: {stats['total']}\n")
            f.write(f"Successful: {stats['success']}\n")
            f.write(f"Skipped (too long): {stats['skipped']}\n")
            f.write(f"Errors: {stats['error']}\n")
            if all_embeddings:
                f.write(f"Embedding matrix shape: {embeddings_matrix.shape}\n")
                f.write(f"Embedding dimension: {embeddings_matrix.shape[1]}\n")
                f.write(f"Number of genes embedded: {embeddings_matrix.shape[0]}\n")
        
        skipped_metadata = self.metadata.loc[skipped_indices].copy() if skipped_indices else pd.DataFrame()
        sequence_id_column = (
            'sequence_id' if 'sequence_id' in self.metadata.columns
            else 'transcript_id' if 'transcript_id' in self.metadata.columns
            else 'gene_id'
        )
        with open(skipped_file, 'w') as f:
            if skipped_metadata.empty:
                f.write("")
            else:
                for _, row in skipped_metadata.iterrows():
                    sequence_id = row.get(sequence_id_column, row.get('gene_id', ''))
                    f.write(
                        f"{sequence_id}\t{row.get('gene_name', 'NA')}\t{row.get('sequence_length_nt', 'NA')}\n"
                    )
        logger.info(f"Saved {len(skipped_indices)} skipped sequences to {skipped_file}")
        
        logger.info(f"\nEmbedding generation complete:")
        logger.info(f"  Total: {stats['total']}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        logger.info(f"  Errors: {stats['error']}")
        
        return stats


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file (config_mane.yaml)."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def resolve_paths(config_dir: Path, config: Dict) -> Dict:
    """Resolve relative paths in config to be relative to config file location."""
    for section in ['input', 'output']:
        if section in config.get('paths', {}):
            for key, path in config['paths'][section].items():
                if path and not Path(path).is_absolute():
                    # Resolve relative to config directory and normalize the path
                    resolved = (config_dir / path).resolve()
                    config['paths'][section][key] = str(resolved)
    
    return config


def main(config_path: str = None):
    """
    Generate embeddings for MANE cDNA sequences using YAML config.
    
    Args:
        config_path: Path to YAML config file (e.g., config/config_mane.yaml)
    """
    parser = argparse.ArgumentParser(
        description="Phase 4: Generate embeddings for MANE canonical transcripts (cDNA)"
    )
    parser.add_argument(
        "--config",
        required=False,
        default=config_path or "config/config_mane.yaml",
        help="Path to YAML config file (default: config/config_mane.yaml)"
    )
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    config = resolve_paths(Path(args.config).parent, config)
    
    # Get paths and settings from config
    output_paths = config['paths']['output']
    embedding_config = config.get('embedding', {})
    
    fasta_path = output_paths.get('extracted_sequences_path')
    manifest_path = output_paths.get('manifest_path')
    output_dir = output_paths.get('reports_dir')
    model_name = config.get('model_name', 'NTv3_650M_pre')
    max_tokens = embedding_config.get('max_tokens_per_batch', 120_000)
    max_sequence_length = embedding_config.get('max_sequence_length', 120_000)
    use_bfloat16 = embedding_config.get('use_bfloat16', True)
    debug = embedding_config.get('debug', False)
    
    # Get output file paths from config
    embeddings_path = output_paths.get('embeddings_path')
    ids_path = output_paths.get('ids_path')
    
    logger.info(f"Configuration loaded:")
    logger.info(f"  FASTA: {fasta_path}")
    logger.info(f"  Manifest: {manifest_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Embeddings path: {embeddings_path}")
    logger.info(f"  IDs path: {ids_path}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Max tokens per batch: {max_tokens:,}")
    logger.info(f"  Max sequence length: {max_sequence_length:,} nt")
    logger.info(f"  Use bfloat16: {use_bfloat16}")
    logger.info(f"  Debug first gene: {debug}")
    
    pipeline = EmbeddingPipeline(
        fasta_path=fasta_path,
        metadata_path=manifest_path,
        output_dir=output_dir,
        embeddings_path=embeddings_path,
        ids_path=ids_path,
        model_name=model_name,
        max_tokens_per_batch=max_tokens,
        max_sequence_length_nt=max_sequence_length,
        use_bfloat16=use_bfloat16,
        debug=debug
    )

    pipeline.run()


if __name__ == '__main__':
    main()
