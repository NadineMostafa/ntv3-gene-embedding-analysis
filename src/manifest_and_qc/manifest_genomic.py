"""
Build genomic sequence manifest and perform QC on extracted sequences.

This module combines:
1. Build symbol-to-genomic-coordinates mapping from GTF
2. Extract genomic sequences from genome FASTA by coordinates
3. Apply QC filters to extracted sequences
4. Generate manifest and failure reports

Input:
  - gene_list.txt: gene symbols to process
  - GTF file: GENCODE/Ensembl GTF annotation
  - Genome FASTA: reference genome (e.g., GRCh38)

Output:
  - manifest_genomic.tsv: genes passing all QC checks
  - genomic_extraction_failures.tsv: genes that failed extraction or QC
  - genomic_qc_summary.txt: statistics and breakdown
"""

import sys
import logging
import pandas as pd
import yaml
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional

try:
    import pyfaidx
except ImportError:
    print("ERROR: pyfaidx not installed. Install with: pip install pyfaidx")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml not installed. Install with: pip install pyyaml")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Phase 1: Parse GTF and build coordinate mapping
# ============================================================================

def parse_gtf(gtf_path: str) -> list:
    """
    Parse GTF file and extract gene-level information.
    
    Only processes 'gene' features to get canonical coordinates.
    
    Args:
        gtf_path: Path to GTF file
        
    Returns:
        list of dicts with keys: gene_id, gene_name, chromosome, start_bp, end_bp, strand
    """
    genes = []
    seen = set()
    
    logger.info(f"Parsing GTF: {gtf_path}")
    with open(gtf_path, 'r') as f:
        line_num = 0
        for line in f:
            line_num += 1
            # Skip comments and empty lines
            if line.startswith('#') or line.strip() == '':
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            feature_type = fields[2]
            
            # Only process gene features
            if feature_type != 'gene':
                continue
            
            # Parse attributes: format is key "value"; key2 "value2"
            attributes = {}
            attr_field = fields[8]
            for match in re.finditer(r'(\w+)\s+"([^"]*)"', attr_field):
                key, val = match.groups()
                attributes[key] = val
            
            # Extract required fields
            gene_id = attributes.get('gene_id')
            gene_name = attributes.get('gene_name')
            
            # Skip if missing critical fields
            if not gene_id or not gene_name:
                continue
            
            # Skip duplicates (GTF may have partial duplicates)
            entry_key = (gene_id, gene_name)
            if entry_key in seen:
                continue
            seen.add(entry_key)
            
            genes.append({
                'gene_id': gene_id,
                'gene_name': gene_name,
                'chromosome': fields[0],
                'start_bp': int(fields[3]),
                'end_bp': int(fields[4]),
                'strand': fields[6]
            })
            
            if line_num % 100000 == 0:
                logger.info(f"  Processed {line_num} lines...")
    
    logger.info(f"  Found {len(genes)} gene features")
    return genes


def build_symbol_coordinate_map(genes: list) -> Tuple[Dict, pd.DataFrame]:
    """
    Map gene symbols to genomic coordinates, resolving multi-mapped genes.
    
    For multi-mapped genes, deterministically selects the longest locus.
    
    Args:
        genes: List of gene dicts from parse_gtf()
        
    Returns:
        (resolved_map, multi_mapped_df)
        - resolved_map: dict {gene_name -> {gene_id, chromosome, start_bp, end_bp, strand, length_bp}}
        - multi_mapped_df: DataFrame of genes with >1 coordinate
    """
    # Group by gene_name
    by_name = defaultdict(list)
    for gene in genes:
        by_name[gene['gene_name']].append(gene)
    
    resolved_map = {}
    multi_mapped = []
    
    logger.info(f"Building symbol-to-coordinate map for {len(by_name)} unique genes...")
    
    for gene_name, entries in by_name.items():
        if len(entries) == 1:
            # Simple case: one coordinate per gene
            entry = entries[0]
            entry['length_bp'] = entry['end_bp'] - entry['start_bp'] + 1
            entry['selection_method'] = 'single'
            resolved_map[gene_name] = entry
        else:
            # Multi-mapped: pick longest locus (deterministic)
            longest = max(entries, key=lambda e: e['end_bp'] - e['start_bp'])
            longest['length_bp'] = longest['end_bp'] - longest['start_bp'] + 1
            longest['selection_method'] = 'multi_mapped_longest'
            resolved_map[gene_name] = longest
            
            # Log conflict
            multi_mapped.append({
                'gene_name': gene_name,
                'n_entries': len(entries),
                'selected_gene_id': longest['gene_id'],
                'selected_length_bp': longest['length_bp'],
                'all_gene_ids': '; '.join([e['gene_id'] for e in entries]),
                'all_lengths_bp': '; '.join([str(e['end_bp'] - e['start_bp'] + 1) for e in entries])
            })
    
    multi_mapped_df = pd.DataFrame(multi_mapped) if multi_mapped else pd.DataFrame()
    logger.info(f"  Resolved {len(resolved_map)} unique symbols")
    if len(multi_mapped_df) > 0:
        logger.info(f"  Multi-mapped: {len(multi_mapped_df)}")
    
    return resolved_map, multi_mapped_df


def load_input_symbols(symbol_file: str) -> set:
    """
    Load input gene symbols from text file.
    
    Handles both single-column and tab-separated formats.
    
    Args:
        symbol_file: Path to file with gene symbols
        
    Returns:
        set of unique gene symbols
    """
    symbols = set()
    logger.info(f"Loading input symbols: {symbol_file}")
    
    with open(symbol_file, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Skip header row
            if i == 0 and line.lower().startswith(('gene_name', 'symbol', 'header', 'gene')):
                continue
            # Extract first column
            symbol = line.split()[0] if line.split() else ''
            if symbol:
                symbols.add(symbol)
    
    logger.info(f"  Loaded {len(symbols)} symbols")
    return symbols


def match_input_to_map(input_symbols: set, resolved_map: Dict) -> Tuple[pd.DataFrame, list]:
    """
    Match input gene symbols against coordinate mapping.
    
    Args:
        input_symbols: Set of input gene symbols
        resolved_map: Resolved symbol -> coordinates dict
        
    Returns:
        (matched_df, unresolved_symbols)
    """
    matched = []
    unresolved = []
    
    logger.info(f"Matching {len(input_symbols)} symbols against map...")
    
    for symbol in sorted(input_symbols):
        if symbol in resolved_map:
            entry = resolved_map[symbol]
            matched.append({
                'gene_name': symbol,
                'gene_id': entry['gene_id'],
                'chromosome': entry['chromosome'],
                'start_bp': entry['start_bp'],
                'end_bp': entry['end_bp'],
                'strand': entry['strand'],
                'length_bp': entry['length_bp'],
                'selection_method': entry.get('selection_method', 'unknown'),
                'status': 'resolved'
            })
        else:
            unresolved.append(symbol)
    
    matched_df = pd.DataFrame(matched)
    logger.info(f"  Matched: {len(matched)}, Unresolved: {len(unresolved)}")
    
    return matched_df, unresolved


# ============================================================================
# Phase 2: Extract genomic sequences and apply QC
# ============================================================================

def reverse_complement(seq: str) -> str:
    """
    Reverse complement a DNA sequence.
    
    Args:
        seq: DNA sequence
        
    Returns:
        Reverse complement sequence
    """
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


class SequenceQC:
    """Quality control checks for genomic sequences."""
    
    def __init__(self, min_length: int = 100, max_length: int = 500000):
        """Initialize QC parameters."""
        self.min_length = min_length
        self.max_length = max_length
        self.valid_bases = set('ACGTN')
    
    def check_sequence(self, seq: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Perform QC checks on a sequence.
        
        Args:
            seq: DNA sequence
        
        Returns:
            (is_valid, normalized_seq, qc_flags)
            - is_valid: True if passes QC
            - normalized_seq: uppercase sequence or None if invalid
            - qc_flags: warning flags or None
        """
        # Check if empty
        if not seq or len(seq.strip()) == 0:
            return False, None, "empty_sequence"
        
        # Normalize to uppercase
        seq_upper = seq.upper().strip()
        
        # Check for invalid bases
        invalid_bases = set(seq_upper) - self.valid_bases
        if invalid_bases:
            return False, None, f"invalid_bases:{','.join(sorted(invalid_bases))}"
        
        # Re-check after normalization
        if len(seq_upper) == 0:
            return False, None, "empty_after_normalization"
        
        # Check length limits
        flags = []
        if len(seq_upper) < self.min_length:
            flags.append(f"too_short:{len(seq_upper)}bp")
        if len(seq_upper) > self.max_length:
            flags.append(f"too_long:{len(seq_upper)}bp")
        
        # Check for too many N's (flag but don't fail)
        n_count = seq_upper.count('N')
        if n_count > len(seq_upper) * 0.5:
            flags.append(f"too_many_Ns:{n_count}/{len(seq_upper)}")
        
        if flags:
            return True, seq_upper, ";".join(flags)  # Pass but flag
        
        return True, seq_upper, None


class GenomicManifestBuilder:
    """Build genomic sequence manifest with QC."""
    
    def __init__(
        self,
        fasta_path: Path,
        gene_list_path: Path,
        gtf_path: Path,
        output_dir: Path,
        min_length: int = 100,
        max_length: int = 500000
    ):
        """Initialize builder."""
        self.fasta_path = Path(fasta_path)
        self.gene_list_path = Path(gene_list_path)
        self.gtf_path = Path(gtf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.qc = SequenceQC(min_length=min_length, max_length=max_length)
        self.fasta = None
    
    def load_fasta(self):
        """Load genome FASTA."""
        logger.info(f"Loading genome FASTA: {self.fasta_path}")
        try:
            self.fasta = pyfaidx.Fasta(str(self.fasta_path), build_index=True)
            logger.info(f"  FASTA loaded with {len(self.fasta)} sequences")
        except Exception as e:
            logger.error(f"Failed to load FASTA: {e}")
            raise RuntimeError(f"Failed to load FASTA: {e}")
    
    def extract_sequence(
        self,
        chromosome: str,
        start_bp: int,
        end_bp: int,
        strand: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract sequence from FASTA by genomic coordinates (1-based inclusive).
        
        Args:
            chromosome: Chromosome name (e.g., 'chr1')
            start_bp: 1-based start coordinate (inclusive)
            end_bp: 1-based end coordinate (inclusive)
            strand: '+' or '-'
        
        Returns:
            (sequence, error_reason)
        """
        if not self.fasta:
            raise RuntimeError("FASTA not loaded")
        
        try:
            # pyfaidx uses 0-based Python slicing
            # GTF is 1-based inclusive: start=100, end=200 means bases 100-200 (101 bases)
            # Python slice: [start-1:end] = [99:200] (also 101 bases)
            seq = str(self.fasta[chromosome][start_bp-1:end_bp].seq)
            
            if not seq:
                return None, "empty_sequence_extracted"
            
            # Reverse complement if on minus strand
            if strand == '-':
                seq = reverse_complement(seq)
            
            return seq, None
        
        except KeyError:
            return None, f"chromosome_not_found:{chromosome}"
        except Exception as e:
            return None, f"extraction_error:{str(e)}"
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame]:
        """
        Process all genes: build mapping, extract sequences, apply QC.
        
        Returns:
            (manifest_df, failures_df, stats, multi_mapped_df, unresolved_df)
        """
        # Parse GTF
        logger.info("Phase 1: Parsing GTF and building coordinate map...")
        genes = parse_gtf(str(self.gtf_path))
        
        # Build mapping
        resolved_map, multi_mapped_df = build_symbol_coordinate_map(genes)
        
        # Load input symbols
        input_symbols = load_input_symbols(str(self.gene_list_path))
        
        # Match to map
        matched_df, unresolved = match_input_to_map(input_symbols, resolved_map)
        
        logger.info(f"Phase 1: {len(matched_df)} genes resolved for extraction")
        
        # Load FASTA
        logger.info("Phase 2: Loading genome FASTA...")
        self.load_fasta()
        
        # Extract and QC
        logger.info("Phase 2: Extracting sequences and applying QC...")
        manifest_records = []
        failure_records = []
        error_counts = defaultdict(int)
        sequences_to_save = []  # Collect sequences that pass QC
        
        for idx, row in matched_df.iterrows():
            gene_name = row['gene_name']
            gene_id = row['gene_id']
            chromosome = row['chromosome']
            start_bp = int(row['start_bp'])
            end_bp = int(row['end_bp'])
            strand = row['strand']
            selection_method = row['selection_method']
            
            # Extract sequence
            seq, extract_error = self.extract_sequence(chromosome, start_bp, end_bp, strand)
            
            if extract_error:
                error_counts[extract_error] += 1
                failure_records.append({
                    'gene_name': gene_name,
                    'gene_id': gene_id,
                    'chromosome': chromosome,
                    'start_bp': start_bp,
                    'end_bp': end_bp,
                    'strand': strand,
                    'selection_method': selection_method,
                    'failure_reason': extract_error,
                    'status': 'failed_extraction'
                })
                if (idx + 1) % 1000 == 0:
                    logger.info(f"  Processed {idx + 1} genes: {len(manifest_records)} passed, {len(failure_records)} failed")
                continue
            
            # Apply QC
            is_valid, norm_seq, qc_flags = self.qc.check_sequence(seq)
            
            if not is_valid:
                failure_records.append({
                    'gene_name': gene_name,
                    'gene_id': gene_id,
                    'chromosome': chromosome,
                    'start_bp': start_bp,
                    'end_bp': end_bp,
                    'strand': strand,
                    'selection_method': selection_method,
                    'failure_reason': qc_flags,
                    'status': 'failed_qc'
                })
                if (idx + 1) % 1000 == 0:
                    logger.info(f"  Processed {idx + 1} genes: {len(manifest_records)} passed, {len(failure_records)} failed")
                continue
            
            # Passed QC - save sequence
            manifest_records.append({
                'gene_name': gene_name,
                'gene_id': gene_id,
                'chromosome': chromosome,
                'start_bp': start_bp,
                'end_bp': end_bp,
                'strand': strand,
                'selection_method': selection_method,
                'sequence_length_bp': len(norm_seq),
                'qc_flags': qc_flags if qc_flags else 'none',
                'status': 'passed_qc'
            })
            
            # Collect sequence for FASTA output
            sequences_to_save.append({
                'header': f"{gene_name}|{gene_id}|{chromosome}:{start_bp}-{end_bp}({strand})",
                'sequence': norm_seq,
                'gene_name': gene_name,
                'chromosome': chromosome,
                'start_bp': start_bp,
                'end_bp': end_bp,
                'strand': strand,
                'length': len(norm_seq)
            })
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Processed {idx + 1} genes: {len(manifest_records)} passed, {len(failure_records)} failed")
        
        manifest_df = pd.DataFrame(manifest_records)
        failures_df = pd.DataFrame(failure_records)
        
        logger.info(f"Phase 2: Extraction complete:")
        logger.info(f"  Passed QC: {len(manifest_df)}")
        logger.info(f"  Failed: {len(failures_df)}")
        logger.info(f"  Failure breakdown:")
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {error_type}: {count}")
        
        # Compile stats
        stats = {
            'total_input': len(input_symbols),
            'resolved': len(matched_df),
            'unresolved': len(unresolved),
            'multi_mapped': len(multi_mapped_df),
            'extracted': len(manifest_df),
            'extraction_failed': len(failures_df),
        }
        
        if len(failures_df) > 0:
            logger.info(f"  Failure breakdown:")
            for status, count in failures_df['status'].value_counts().items():
                logger.info(f"    {status}: {count}")
        
        # Create unresolved dataframe
        unresolved_df = pd.DataFrame({
            'gene_name': unresolved,
            'status': 'unresolved_in_gtf'
        }) if unresolved else pd.DataFrame()
        
        return manifest_df, failures_df, stats, multi_mapped_df, unresolved_df, sequences_to_save
    
    def save_results(self, manifest_df: pd.DataFrame, failures_df: pd.DataFrame, stats: Dict, 
                     multi_mapped_df: pd.DataFrame = None, unresolved_df: pd.DataFrame = None, sequences: list = None):
        """Save manifest, failures, multi-mapped, unresolved, sequences FASTA, and summary."""
        # Save manifest
        manifest_path = self.output_dir / 'manifest_genomic.tsv'
        manifest_df.to_csv(manifest_path, sep='\t', index=False)
        logger.info(f"Saved manifest to {manifest_path}")
        
        # Save failures
        failures_path = self.output_dir / 'genomic_extraction_failures.tsv'
        failures_df.to_csv(failures_path, sep='\t', index=False)
        logger.info(f"Saved failures to {failures_path}")
        
        # Save multi-mapped if present
        if multi_mapped_df is not None and len(multi_mapped_df) > 0:
            multi_path = self.output_dir / 'multi_mapped_genes.tsv'
            multi_mapped_df.to_csv(multi_path, sep='\t', index=False)
            logger.info(f"Saved multi-mapped genes to {multi_path}")
        
        # Save unresolved (always save, even if empty)
        unresolved_path = self.output_dir / 'unresolved_genes.tsv'
        if unresolved_df is not None:
            unresolved_df.to_csv(unresolved_path, sep='\t', index=False)
            logger.info(f"Saved unresolved genes to {unresolved_path} ({len(unresolved_df)} genes)")
        else:
            # Create empty file with headers if no unresolved
            pd.DataFrame(columns=['gene_name', 'gene_id', 'chromosome', 'start_bp', 'end_bp', 'status']).to_csv(
                unresolved_path, sep='\t', index=False)
            logger.info(f"Saved empty unresolved genes file to {unresolved_path}")
        
        # Save FASTA sequences
        if sequences and len(sequences) > 0:
            fasta_path = self.output_dir / 'genomic_sequences.fasta'
            logger.info(f"Saving {len(sequences)} sequences to FASTA...")
            with open(fasta_path, 'w') as f:
                for entry in sequences:
                    f.write(f">{entry['header']}\n")
                    # Write sequence in 80-character lines
                    seq = entry['sequence']
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + '\n')
            logger.info(f"Saved FASTA sequences to {fasta_path}")
            
            # Save sequence lengths metadata
            lengths_path = self.output_dir / 'genomic_sequence_lengths.tsv'
            lengths_df = pd.DataFrame([
                {
                    'gene_name': entry['gene_name'],
                    'chromosome': entry['chromosome'],
                    'start_bp': entry['start_bp'],
                    'end_bp': entry['end_bp'],
                    'strand': entry['strand'],
                    'sequence_length_bp': entry['length']
                }
                for entry in sequences
            ])
            lengths_df.to_csv(lengths_path, sep='\t', index=False)
            logger.info(f"Saved sequence lengths to {lengths_path}")
        
        # Save summary
        summary_path = self.output_dir / 'genomic_qc_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GENOMIC SEQUENCE MANIFEST & QC SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("[RESOLUTION PHASE]\n")
            f.write(f"  Input symbols:            {stats['total_input']:,}\n")
            f.write(f"  Resolved:                 {stats['resolved']:,} ({100*stats['resolved']/max(stats['total_input'],1):.1f}%)\n")
            f.write(f"  Unresolved:               {stats['unresolved']:,}\n")
            f.write(f"  Multi-mapped:             {stats['multi_mapped']:,}\n")
            
            f.write(f"\n[EXTRACTION & QC PHASE]\n")
            f.write(f"  Extraction attempted:     {stats['resolved']:,}\n")
            f.write(f"  Extraction succeeded:     {stats['extracted']:,}\n")
            f.write(f"  Extraction failed:        {stats['extraction_failed']:,}\n")
            
            if len(manifest_df) > 0:
                f.write(f"\n[SEQUENCE STATISTICS]\n")
                f.write(f"  Length (bp) - Min:        {manifest_df['sequence_length_bp'].min():,}\n")
                f.write(f"  Length (bp) - Max:        {manifest_df['sequence_length_bp'].max():,}\n")
                f.write(f"  Length (bp) - Mean:       {manifest_df['sequence_length_bp'].mean():,.0f}\n")
                f.write(f"  Length (bp) - Median:     {manifest_df['sequence_length_bp'].median():,.0f}\n")
                
                f.write(f"\n[QC FLAGS]\n")
                for flag, count in manifest_df['qc_flags'].value_counts().items():
                    f.write(f"  {flag}: {count}\n")
            
            if len(failures_df) > 0:
                f.write(f"\n[FAILURE BREAKDOWN]\n")
                for status, count in failures_df['status'].value_counts().items():
                    f.write(f"  {status}: {count}\n")
                
                qc_failures = failures_df[failures_df['status'] == 'failed_qc']
                if len(qc_failures) > 0:
                    f.write(f"\n[QC FAILURE REASONS]\n")
                    for reason, count in qc_failures['failure_reason'].value_counts().items():
                        f.write(f"  {reason}: {count}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Saved summary to {summary_path}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        dict: Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading config: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def resolve_paths(config_dir: Path, config: Dict) -> Dict:
    """
    Resolve relative paths in config to be relative to config file location.
    
    Args:
        config_dir: Directory containing the config file
        config: Configuration dictionary
        
    Returns:
        dict: Configuration with resolved paths
    """
    for section in ['input', 'output']:
        if section in config.get('paths', {}):
            for key, path_val in config['paths'][section].items():
                if isinstance(path_val, str) and path_val.startswith('../'):
                    # Resolve relative to config directory
                    resolved = (config_dir / path_val).resolve()
                    config['paths'][section][key] = str(resolved)
    
    return config


def main(config_path: str):
    """
    Build genomic sequence manifest from YAML config.
    
    Args:
        config_path: Path to YAML config file (e.g., config/config_genomic.yaml)
        
    Returns:
        manifest_df: DataFrame with passing sequences
        failures_df: DataFrame with failed sequences
        stats: Dictionary with QC statistics
    """
    config = load_config(config_path)
    config = resolve_paths(Path(config_path).parent, config)
    
    paths = config['paths']['input']
    output_paths = config['paths']['output']
    embedding = config.get('embedding', {})
    
    fasta_path = paths['genome_path']
    gene_list_path = paths['gene_list_path']
    gtf_path = paths['annotation_path']
    output_dir = output_paths['reports_dir']
    max_length = embedding.get('max_length', 500000)
    
    builder = GenomicManifestBuilder(
        fasta_path=fasta_path,
        gene_list_path=gene_list_path,
        gtf_path=gtf_path,
        output_dir=output_dir,
        max_length=max_length
    )
    
    manifest_df, failures_df, stats, multi_mapped_df, unresolved_df, sequences = builder.process()
    builder.save_results(manifest_df, failures_df, stats, multi_mapped_df, unresolved_df, sequences)
    
    logger.info("Genomic manifest generation complete!")
    return manifest_df, failures_df, stats


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python manifest_genomic.py <config.yaml>")
        print("\nExample:")
        print("  python manifest_genomic.py ../../config/config_genomic.yaml")
        sys.exit(1)
    
    main(config_path=sys.argv[1])
