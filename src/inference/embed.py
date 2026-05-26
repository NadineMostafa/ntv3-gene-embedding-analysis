#!/usr/bin/env python3
"""
Unified embedding pipeline for MANE transcript and genomic sequences.

Configuration-driven using config_mane.yaml or config_genomic.yaml.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyfaidx
import yaml
from nucleotide_transformer_v3.pretrained import get_pretrained_ntv3_model
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

jax.config.update("jax_platform_name", "gpu")


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_paths(config_dir: Path, config: Dict) -> Dict:
    """Resolve relative config paths against the config file location."""
    for section in ["input", "output"]:
        if section in config.get("paths", {}):
            for key, value in config["paths"][section].items():
                if value and not Path(value).is_absolute():
                    config["paths"][section][key] = str((config_dir / value).resolve())
    return config


class AdaptiveBatcher:
    """Group sequences into batches based on sequence length."""

    def __init__(self, max_tokens_per_batch: int, pad_multiple: int):
        self.max_tokens_per_batch = max_tokens_per_batch
        self.pad_multiple = max(1, pad_multiple)

    def _round_up_to_pad_multiple(self, sequence_length: int) -> int:
        """Round sequence length up to the model padding multiple."""
        return (
            (sequence_length + self.pad_multiple - 1) // self.pad_multiple
        ) * self.pad_multiple

    def create_batches(
        self,
        metadata: pd.DataFrame,
        length_column: str,
        max_sequence_length: int,
    ) -> Tuple[List[List[int]], List[int]]:
        """Create batches and filter oversized sequences before inference."""
        if length_column not in metadata.columns:
            raise ValueError(f"Metadata is missing required length column: {length_column}")

        valid_metadata = metadata[metadata[length_column] <= max_sequence_length]
        skipped_indices = metadata[metadata[length_column] > max_sequence_length].index.tolist()
        sorted_idx = valid_metadata.sort_values(by=length_column).index.tolist()

        batches: List[List[int]] = []
        current_batch: List[int] = []

        for idx in sorted_idx:
            longest_sequence_length = int(metadata.loc[idx, length_column])
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


class NTv3Embedder:
    """Generate embeddings using NTv3 with JAX/Flax."""

    def __init__(
        self,
        fasta_path: str,
        model_name: str,
        use_bfloat16: bool,
        mode: str,
    ):
        logger.info(f"Loading model: {model_name} (bfloat16={use_bfloat16}, mode={mode})")

        logger.info("\n" + "=" * 60)
        logger.info("GPU CONFIGURATION CHECK")
        logger.info("=" * 60)

        devices = jax.devices()
        gpu_devices = [device for device in devices if "gpu" in device.platform.lower()]

        logger.info(f"JAX Platform Name: {jax.devices()[0].platform}")
        logger.info(f"Total JAX devices: {len(devices)}")
        logger.info(f"Available devices: {devices}")
        logger.info(f"GPU devices found: {len(gpu_devices)}")

        if gpu_devices:
            logger.info("GPU is enabled and available")
            for idx, device in enumerate(gpu_devices):
                logger.info(f"  GPU {idx}: {device}")
        else:
            logger.warning("GPU is not available - falling back to CPU")

        logger.info("=" * 60 + "\n")

        self.pretrained_model, self.tokenizer, self.config = get_pretrained_ntv3_model(
            model_name=model_name,
            embeddings_layers_to_save=(6,),
            attention_maps_to_save=((6, 1),),
            use_bfloat16=use_bfloat16,
        )

        self.fasta = pyfaidx.Fasta(fasta_path)
        logger.info(f"Loaded FASTA file: {fasta_path}")
        logger.info(f"Number of sequences in FASTA: {len(self.fasta.keys())}")

        logger.info("\nTesting GPU with small computation...")
        try:
            test_arr = jnp.ones((100, 100))
            result = jnp.dot(test_arr, test_arr)
            logger.info("GPU test computation successful")
            logger.info(f"  Result shape: {result.shape}, device: {result.devices()}")
        except Exception as exc:
            logger.warning(f"GPU test failed: {exc}")

    def get_sequence(self, sequence_id: str) -> str:
        """Load sequence from FASTA by exact or substring ID match."""
        fasta_key = None

        if sequence_id in self.fasta:
            fasta_key = sequence_id
        else:
            for key in self.fasta.keys():
                if sequence_id in key or key in sequence_id:
                    fasta_key = key
                    break

        if fasta_key is None:
            raise ValueError(
                f"Sequence ID {sequence_id} not found in FASTA. "
                f"Available keys: {list(self.fasta.keys())[:5]}..."
            )

        return str(self.fasta[fasta_key][:].seq).upper()

    def embed_batch(self, sequence_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Generate embeddings for a batch of sequences."""
        sequences = []
        for sequence_id in sequence_ids:
            sequence = self.get_sequence(sequence_id)
            logger.info(
                f"Processing sequence {sequence_id} with raw length {len(sequence):,} bases"
            )
            sequences.append(sequence)

        if not sequences:
            return np.array([]), []

        tokens = self.tokenizer.batch_np_tokenize(sequences)

        num_downsamples = self.config.num_downsamples
        pad_multiple = 2 ** num_downsamples

        _, seq_length = tokens.shape
        padded_length = ((seq_length + pad_multiple - 1) // pad_multiple) * pad_multiple

        if padded_length > seq_length:
            padding = padded_length - seq_length
            tokens = np.pad(
                tokens,
                ((0, 0), (0, padding)),
                mode="constant",
                constant_values=self.tokenizer.pad_token_id,
            )
            logger.debug(
                f"Padded tokens from {seq_length} to {padded_length} (multiple of {pad_multiple})"
            )

        logger.info(
            f"Running model on token batch shape {tokens.shape} "
            f"(unpadded_length={seq_length:,}, padded_length={padded_length:,})"
        )

        outs = self.pretrained_model(tokens)
        final_embeddings = outs["embedding"]

        padding_mask = jnp.expand_dims(tokens != self.tokenizer.pad_token_id, axis=-1)
        masked_embeddings = final_embeddings * padding_mask
        sequence_lengths = jnp.sum(padding_mask, axis=1)
        mean_embeddings = jnp.sum(masked_embeddings, axis=1) / sequence_lengths

        return np.array(mean_embeddings), sequence_ids


class EmbeddingPipeline:
    """Unified embedding pipeline for either MANE or genomic mode."""

    def __init__(
        self,
        mode: str,
        fasta_path: str,
        metadata_path: str,
        output_dir: str,
        embeddings_path: str,
        ids_path: str,
        model_name: str,
        sequence_id_column: str,
        length_column: str,
        max_tokens_per_batch: int,
        max_sequence_length: int,
        use_bfloat16: bool,
        debug: bool,
    ):
        self.mode = mode
        self.fasta_path = fasta_path
        self.metadata_path = metadata_path
        self.output_dir = Path(output_dir)
        self.embeddings_path = Path(embeddings_path)
        self.ids_path = Path(ids_path)
        self.sequence_id_column = sequence_id_column
        self.length_column = length_column
        self.max_sequence_length = max_sequence_length
        self.debug = debug

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = pd.read_csv(metadata_path, sep="\t")
        logger.info(f"Loaded metadata for {len(self.metadata)} sequences")
        logger.info(f"Metadata columns: {list(self.metadata.columns)}")

        if self.sequence_id_column not in self.metadata.columns:
            raise ValueError(f"Metadata is missing required ID column: {self.sequence_id_column}")
        if self.length_column not in self.metadata.columns:
            raise ValueError(f"Metadata is missing required length column: {self.length_column}")

        self.embedder = NTv3Embedder(
            fasta_path=fasta_path,
            model_name=model_name,
            use_bfloat16=use_bfloat16,
            mode=mode,
        )
        self.batcher = AdaptiveBatcher(
            max_tokens_per_batch=max_tokens_per_batch,
            pad_multiple=2 ** self.embedder.config.num_downsamples,
        )

        logger.info(f"Adaptive batcher configured with max_tokens_per_batch={max_tokens_per_batch:,}")
        logger.info(f"Max sequence length: {max_sequence_length:,} bases")

    def debug_first_sequence(self) -> None:
        """Show first-sequence transformation for debugging."""
        if not self.debug or len(self.metadata) == 0:
            return

        logger.info("\n" + "=" * 70)
        logger.info(f"DEBUG: First Sequence Transformation ({self.mode})")
        logger.info("=" * 70)

        first_row = self.metadata.iloc[0]
        first_seq_id = first_row[self.sequence_id_column]
        sequence = self.embedder.get_sequence(first_seq_id)

        logger.info("\n1. METADATA:")
        logger.info(f"   Sequence ID: {first_seq_id}")
        logger.info(f"   Gene Symbol: {first_row.get('gene_name', 'N/A')}")
        logger.info(f"   Gene ID: {first_row.get('gene_id', 'N/A')}")
        logger.info(f"   Length: {first_row.get(self.length_column, 'N/A')}")

        logger.info("\n2. SEQUENCE:")
        logger.info(f"   Full length: {len(sequence)} bases")
        logger.info(f"   First 100 bases: {sequence[:100]}")
        if len(sequence) > 100:
            logger.info(f"   Last 100 bases: {sequence[-100:]}")

        if len(sequence) > self.max_sequence_length:
            logger.info(
                f"   Embedding skipped in debug: sequence length exceeds max "
                f"{self.max_sequence_length:,} bases"
            )
            logger.info("=" * 70 + "\n")
            return

        logger.info("\n3. TOKENIZATION:")
        tokens = self.embedder.tokenizer.batch_np_tokenize([sequence])
        token_ids = tokens[0].tolist()
        logger.info(f"   Sequence length: {len(sequence)} bases")
        logger.info(f"   Token length (padded): {len(token_ids)}")
        logger.info(f"   First 20 tokens: {token_ids[:20]}")
        logger.info("   Token mapping: A=6, T=7, C=8, G=9, N=10, Pad=1")

        logger.info("\n4. EMBEDDING GENERATION:")
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

        logger.info("=" * 70 + "\n")

    def run(self) -> Dict[str, Any]:
        """Run the full embedding pipeline."""
        self.debug_first_sequence()

        batches, skipped_indices = self.batcher.create_batches(
            self.metadata,
            length_column=self.length_column,
            max_sequence_length=self.max_sequence_length,
        )
        logger.info(
            f"Created {len(batches)} adaptive batches "
            f"(max_tokens_per_batch={self.batcher.max_tokens_per_batch:,})"
        )
        logger.info(
            f"Filtered {len(skipped_indices)} sequences longer than "
            f"{self.max_sequence_length:,} bases before inference"
        )

        stats_file = self.output_dir / f"phase4_embedding_stats_{self.mode}.txt"
        skipped_file = self.output_dir / f"skipped_sequences_{self.mode}.txt"

        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        self.ids_path.parent.mkdir(parents=True, exist_ok=True)
        stats_file.parent.mkdir(parents=True, exist_ok=True)

        all_embeddings = []
        all_sequence_ids = []
        stats = {
            "total": len(self.metadata),
            "success": 0,
            "error": 0,
            "skipped": len(skipped_indices),
        }

        for batch_idx, batch_indices in enumerate(
            tqdm(batches, desc="Processing batches", unit="batch")
        ):
            batch_size = len(batch_indices)

            try:
                sequence_ids = self.metadata.loc[batch_indices, self.sequence_id_column].tolist()
                sequence_lengths = self.metadata.loc[batch_indices, self.length_column].sum()

                embeddings, ids = self.embedder.embed_batch(sequence_ids)

                if len(embeddings) > 0:
                    all_embeddings.append(embeddings)
                    all_sequence_ids.extend(ids)
                    stats["success"] += len(ids)
                    logger.debug(
                        f"Batch {batch_idx}: processed {len(ids)} sequences ({sequence_lengths:,} bases)"
                    )
                else:
                    logger.warning(f"Batch {batch_idx}: no embeddings generated")

            except Exception as exc:
                logger.error(f"Error in batch {batch_idx}: {exc}")
                stats["error"] += batch_size

        if all_embeddings:
            embeddings_matrix = np.vstack(all_embeddings)
            logger.info(f"Embedding matrix shape: {embeddings_matrix.shape}")
            np.save(self.embeddings_path, embeddings_matrix)
            logger.info(f"Saved embeddings to {self.embeddings_path}")

            with self.ids_path.open("w", encoding="utf-8") as handle:
                handle.write("\n".join(all_sequence_ids))
            logger.info(f"Saved sequence IDs to {self.ids_path}")
        else:
            embeddings_matrix = np.array([])

        with stats_file.open("w", encoding="utf-8") as handle:
            handle.write(f"Embedding Generation Summary ({self.mode})\n")
            handle.write("=" * 60 + "\n")
            handle.write(f"Input FASTA: {self.fasta_path}\n")
            handle.write(f"Metadata: {self.metadata_path}\n")
            handle.write(f"Mode: {self.mode}\n")
            handle.write("Framework: JAX/Flax\n")
            handle.write("Chunking: DISABLED (full sequence per row)\n")
            handle.write("Pooling: Mean pooling over all tokens\n")
            handle.write(
                f"Batching: Adaptive (max tokens per batch={self.batcher.max_tokens_per_batch:,})\n"
            )
            handle.write(f"Max sequence length: {self.max_sequence_length:,} bases\n")
            handle.write(f"Total sequences: {stats['total']}\n")
            handle.write(f"Successful: {stats['success']}\n")
            handle.write(f"Skipped (too long): {stats['skipped']}\n")
            handle.write(f"Errors: {stats['error']}\n")
            if all_embeddings:
                handle.write(f"Embedding matrix shape: {embeddings_matrix.shape}\n")
                handle.write(f"Embedding dimension: {embeddings_matrix.shape[1]}\n")

        skipped_metadata = self.metadata.loc[skipped_indices].copy() if skipped_indices else pd.DataFrame()
        with skipped_file.open("w", encoding="utf-8") as handle:
            if skipped_metadata.empty:
                handle.write("")
            else:
                for _, row in skipped_metadata.iterrows():
                    handle.write(
                        f"{row.get(self.sequence_id_column, '')}\t"
                        f"{row.get('gene_name', 'NA')}\t"
                        f"{row.get(self.length_column, 'NA')}\n"
                    )
        logger.info(f"Saved {len(skipped_indices)} skipped sequences to {skipped_file}")

        logger.info("\nEmbedding generation complete:")
        logger.info(f"  Total: {stats['total']}")
        logger.info(f"  Success: {stats['success']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        logger.info(f"  Errors: {stats['error']}")

        return {
            "mode": self.mode,
            "total_sequences": stats["total"],
            "successfully_embedded": stats["success"],
            "skipped_sequences": stats["skipped"],
            "failed_batches_count": stats["error"],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings for MANE or genomic sequences")
    parser.add_argument(
        "--config",
        required=False,
        default="config/config_mane.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_paths(Path(args.config).parent, config)

    mode = str(config.get("mode", "mane"))
    output_paths = config["paths"]["output"]
    embedding_config = config.get("embedding", {})

    fasta_path = output_paths["extracted_sequences_path"]
    metadata_path = output_paths["manifest_path"]
    output_dir = output_paths["reports_dir"]
    embeddings_path = output_paths["embeddings_path"]
    ids_path = output_paths["ids_path"]

    model_name = str(config.get("model_name", "NTv3_650M_pre"))
    sequence_id_column = str(embedding_config.get("sequence_id_column", "transcript_id"))
    length_column = str(embedding_config.get("length_column", "sequence_length_bp"))
    max_tokens_per_batch = int(embedding_config.get("max_tokens_per_batch", 120_000))
    max_sequence_length = int(
        embedding_config.get("max_sequence_length", embedding_config.get("max_length", 120_000))
    )
    use_bfloat16 = bool(embedding_config.get("use_bfloat16", True))
    debug = bool(embedding_config.get("debug", False))

    logger.info("Configuration loaded:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  FASTA: {fasta_path}")
    logger.info(f"  Manifest: {metadata_path}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Embeddings path: {embeddings_path}")
    logger.info(f"  IDs path: {ids_path}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Sequence ID column: {sequence_id_column}")
    logger.info(f"  Length column: {length_column}")
    logger.info(f"  Max tokens per batch: {max_tokens_per_batch:,}")
    logger.info(f"  Max sequence length: {max_sequence_length:,} bases")
    logger.info(f"  Use bfloat16: {use_bfloat16}")
    logger.info(f"  Debug first sequence: {debug}")

    pipeline = EmbeddingPipeline(
        mode=mode,
        fasta_path=fasta_path,
        metadata_path=metadata_path,
        output_dir=output_dir,
        embeddings_path=embeddings_path,
        ids_path=ids_path,
        model_name=model_name,
        sequence_id_column=sequence_id_column,
        length_column=length_column,
        max_tokens_per_batch=max_tokens_per_batch,
        max_sequence_length=max_sequence_length,
        use_bfloat16=use_bfloat16,
        debug=debug,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
