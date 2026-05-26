"""
Build MANE transcript manifest and perform QC on cDNA sequences.

This module combines:
1. Build symbol-to-canonical-transcript mapping from GTF
2. Extract cDNA sequences from transcript FASTA
3. Apply QC filters to extracted sequences
4. Generate manifest and failure reports

Input:
  - gene_list.txt: gene symbols to process
  - GTF file: GENCODE/Ensembl GTF annotation
  - Transcript FASTA: cDNA sequences

Output:
  - manifest_mane.tsv: genes passing all QC checks
  - mane_extraction_failures.tsv: genes that failed extraction or QC
  - mane_qc_summary.txt: statistics and breakdown
"""

import sys
import logging
import pandas as pd
import yaml
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
# Phase 1: Parse GTF and build transcript mapping
# ============================================================================

def parse_gtf_attributes(attr_string: str) -> Dict:
    """
    Parse GTF attribute column (column 9).
    
    Handles multiple values for the same key (e.g., multiple tags).
    
    Args:
        attr_string: GTF attribute string
        
    Returns:
        dict: parsed attributes
    """
    attrs = {}
    for item in attr_string.split(';'):
        item = item.strip()
        if not item:
            continue
        if ' ' in item:
            key, val = item.split(' ', 1)
            val = val.strip('"')
            
            # For fields that can appear multiple times (like 'tag'), collect as list
            if key in ['tag']:
                if key not in attrs:
                    attrs[key] = []
                attrs[key].append(val)
            else:
                attrs[key] = val
    return attrs


def parse_gtf(gtf_path: str) -> Dict:
    """
    Parse GTF file and extract gene_name -> transcript relationships.
    
    Args:
        gtf_path: Path to GTF file
        
    Returns:
        dict with keys:
            - 'genes': {gene_name: {gene_id, transcripts: [list]}}
            - 'transcripts': {transcript_id: {metadata}}
    """
    genes = defaultdict(lambda: {'gene_id': None, 'transcripts': []})
    transcripts = {}
    
    logger.info(f"Parsing GTF: {gtf_path}")
    with open(gtf_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#'):
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            
            feature_type = fields[2]
            if feature_type not in ['gene', 'transcript']:
                continue
            
            # Parse attributes
            attrs = parse_gtf_attributes(fields[8])
            
            if feature_type == 'gene':
                gene_name = attrs.get('gene_name')
                gene_id = attrs.get('gene_id')
                if gene_name and gene_id:
                    genes[gene_name]['gene_id'] = gene_id
                    
            elif feature_type == 'transcript':
                gene_name = attrs.get('gene_name')
                gene_id = attrs.get('gene_id')
                transcript_id = attrs.get('transcript_id')
                
                if gene_name and transcript_id:
                    genes[gene_name]['transcripts'].append(transcript_id)
                    
                    # Store transcript metadata
                    transcripts[transcript_id] = {
                        'gene_name': gene_name,
                        'gene_id': gene_id,
                        'tag': attrs.get('tag', []),
                        'transcript_type': attrs.get('transcript_type', 'unknown'),
                        'appris': attrs.get('appris', ''),
                    }
            
            if line_num % 100000 == 0:
                logger.info(f"  Processed {line_num} lines...")
    
    logger.info(f"  Total genes: {len(genes)}, total transcripts: {len(transcripts)}")
    return {'genes': genes, 'transcripts': transcripts}


def select_canonical_transcript(
    transcript_ids: list,
    gtf_data: Dict
) -> Tuple[Optional[str], str]:
    """
    Select canonical transcript using priority rules.
    
    Rules (in priority order):
      1. MANE Select
      2. Canonical tag
      3. APPRIS principal
      4. Protein-coding
      5. First available
    
    Args:
        transcript_ids: list of transcript IDs for a gene
        gtf_data: GTF data dict
        
    Returns:
        (selected_transcript_id, fallback_rule_used)
    """
    if not transcript_ids:
        return None, 'no_transcripts'
    
    transcript_data = gtf_data['transcripts']
    
    # Rule 1: MANE Select
    for tid in transcript_ids:
        if tid in transcript_data:
            tags = transcript_data[tid].get('tag', [])
            if isinstance(tags, list) and 'MANE_Select' in tags:
                return tid, 'MANE_Select'
    
    # Rule 2: Canonical tag
    for tid in transcript_ids:
        if tid in transcript_data:
            tags = transcript_data[tid].get('tag', [])
            if isinstance(tags, list) and 'canonical' in tags:
                return tid, 'canonical_tag'
    
    # Rule 3: APPRIS principal
    for tid in transcript_ids:
        if tid in transcript_data:
            appris = transcript_data[tid].get('appris', '')
            if appris == 'principal':
                return tid, 'appris_principal'
    
    # Rule 4: Protein-coding
    for tid in transcript_ids:
        if tid in transcript_data:
            ttype = transcript_data[tid].get('transcript_type', '')
            if 'protein_coding' in ttype:
                return tid, 'protein_coding'
    
    # Rule 5: First available
    return transcript_ids[0], 'first_available'


def build_symbol_transcript_map(gene_list_path: str, gtf_data: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Build symbol-to-transcript mapping for input genes.
    
    Args:
        gene_list_path: Path to gene list file
        gtf_data: Parsed GTF data
        
    Returns:
        (mapping_df, stats_dict)
    """
    # Load input gene list
    logger.info(f"Loading gene list: {gene_list_path}")
    gene_df = pd.read_csv(gene_list_path, sep='\t', header=None, skiprows=1)
    gene_symbols = gene_df.iloc[:, 0].tolist()
    logger.info(f"  Loaded {len(gene_symbols)} gene symbols")
    
    gtf_genes = gtf_data['genes']
    
    # Build mapping
    mapping_records = []
    stats = {
        'total_input': len(gene_symbols),
        'resolved': 0,
        'unresolved_not_in_gtf': 0,
        'unresolved_no_transcripts': 0,
        'used_mane_select': 0,
        'used_canonical_tag': 0,
        'used_appris_principal': 0,
        'used_protein_coding': 0,
        'used_first_available': 0,
    }
    
    logger.info(f"Building mapping for {len(gene_symbols)} genes...")
    
    for idx, symbol in enumerate(gene_symbols):
        if idx % 1000 == 0 and idx > 0:
            logger.info(f"  Processed {idx}/{len(gene_symbols)} genes...")
        
        if symbol not in gtf_genes:
            mapping_records.append({
                'gene_name': symbol,
                'gene_id': None,
                'transcript_id': None,
                'fallback_rule': None,
                'status': 'unresolved_not_in_gtf'
            })
            stats['unresolved_not_in_gtf'] += 1
            continue
        
        gene_info = gtf_genes[symbol]
        gene_id = gene_info['gene_id']
        transcript_ids = gene_info['transcripts']
        
        if not transcript_ids:
            mapping_records.append({
                'gene_name': symbol,
                'gene_id': gene_id,
                'transcript_id': None,
                'fallback_rule': None,
                'status': 'unresolved_no_transcripts'
            })
            stats['unresolved_no_transcripts'] += 1
            continue
        
        # Select canonical transcript
        selected_tid, fallback_rule = select_canonical_transcript(transcript_ids, gtf_data)
        
        mapping_records.append({
            'gene_name': symbol,
            'gene_id': gene_id,
            'transcript_id': selected_tid,
            'fallback_rule': fallback_rule,
            'status': 'resolved'
        })
        
        stats['resolved'] += 1
        
        # Track fallback rule usage
        if fallback_rule == 'MANE_Select':
            stats['used_mane_select'] += 1
        elif fallback_rule == 'canonical_tag':
            stats['used_canonical_tag'] += 1
        elif fallback_rule == 'appris_principal':
            stats['used_appris_principal'] += 1
        elif fallback_rule == 'protein_coding':
            stats['used_protein_coding'] += 1
        elif fallback_rule == 'first_available':
            stats['used_first_available'] += 1
    
    mapping_df = pd.DataFrame(mapping_records)
    logger.info(f"  Mapping complete!")
    
    return mapping_df, stats


# ============================================================================
# Phase 2: Extract cDNA sequences and apply QC
# ============================================================================

class SequenceQC:
    """Quality control checks for sequences."""
    
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
            flags.append(f"too_short:{len(seq_upper)}nt")
        if len(seq_upper) > self.max_length:
            flags.append(f"too_long:{len(seq_upper)}nt")
        
        if flags:
            return True, seq_upper, ";".join(flags)  # Pass but flag
        
        return True, seq_upper, None


class MANEManifestBuilder:
    """Build MANE transcript manifest with QC."""
    
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
        """Load FASTA and build transcript ID lookup."""
        logger.info(f"Loading FASTA: {self.fasta_path}")
        try:
            self.fasta = pyfaidx.Fasta(str(self.fasta_path), build_index=True)
            logger.info(f"  FASTA loaded with {len(self.fasta)} sequences")
            
            # Build transcript ID lookup (GENCODE format: ENST...|ENSG...|...)
            self.tx_id_to_key = {}
            self.tx_base_to_versioned = defaultdict(list)
            logger.info("  Building transcript ID lookup index...")
            for fasta_key in self.fasta.keys():
                tx_id = fasta_key.split('|')[0]
                self.tx_id_to_key[tx_id] = fasta_key
                # Also index base ID (without version) for fallback matching
                tx_base = tx_id.split('.')[0]
                self.tx_base_to_versioned[tx_base].append(tx_id)
            
            logger.info(f"  Lookup index built with {len(self.tx_id_to_key)} transcripts")
        
        except Exception as e:
            logger.error(f"Failed to load FASTA: {e}")
            raise RuntimeError(f"Failed to load FASTA: {e}")
    
    def extract_sequence(self, transcript_id: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract sequence for a transcript.
        
        Args:
            transcript_id: Ensembl transcript ID
            
        Returns:
            (sequence, error_reason)
        """
        if not self.fasta or not hasattr(self, 'tx_id_to_key'):
            raise RuntimeError("FASTA not loaded")
        
        try:
            # Try exact match first
            if transcript_id in self.tx_id_to_key:
                fasta_key = self.tx_id_to_key[transcript_id]
                seq = str(self.fasta[fasta_key])
                return seq, None
            
            # Try base ID match (without version) using precomputed lookup
            tx_base = transcript_id.split('.')[0]
            if tx_base in self.tx_base_to_versioned:
                # Use first available versioned form
                matched_tx_id = self.tx_base_to_versioned[tx_base][0]
                fasta_key = self.tx_id_to_key[matched_tx_id]
                seq = str(self.fasta[fasta_key])
                return seq, None
            
            return None, f"transcript_not_found:{transcript_id}"
        
        except Exception as e:
            return None, f"extraction_error:{str(e)}"
    
    def process(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, pd.DataFrame]:
        """
        Process all genes: build mapping, extract sequences, apply QC.
        
        Returns:
            (manifest_df, failures_df, stats, unresolved_df)
        """
        # Parse GTF
        logger.info("Phase 1: Parsing GTF and building transcript map...")
        gtf_data = parse_gtf(str(self.gtf_path))
        
        # Build mapping
        mapping_df, mapping_stats = build_symbol_transcript_map(str(self.gene_list_path), gtf_data)
        
        # Filter to resolved genes only
        resolved_df = mapping_df[mapping_df['status'] == 'resolved'].copy()
        logger.info(f"Phase 1: {len(resolved_df)} genes resolved for extraction")
        
        # Load FASTA
        logger.info("Phase 2: Loading FASTA...")
        self.load_fasta()
        
        # Extract and QC
        logger.info("Phase 2: Extracting sequences and applying QC...")
        manifest_records = []
        failure_records = []
        error_counts = defaultdict(int)
        sequences_to_save = []  # Collect sequences that pass QC
        
        for idx, row in resolved_df.iterrows():
            gene_name = row['gene_name']
            gene_id = row['gene_id']
            transcript_id = row['transcript_id']
            fallback_rule = row['fallback_rule']
            
            # Extract sequence
            seq, extract_error = self.extract_sequence(transcript_id)
            
            if extract_error:
                error_counts[extract_error] += 1
                if len(failure_records) == 1:
                    # Log first failure as sample
                    logger.info(f"Sample failure: {transcript_id} -> {extract_error}")
                failure_records.append({
                    'gene_name': gene_name,
                    'gene_id': gene_id,
                    'transcript_id': transcript_id,
                    'fallback_rule': fallback_rule,
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
                    'transcript_id': transcript_id,
                    'fallback_rule': fallback_rule,
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
                'transcript_id': transcript_id,
                'fallback_rule': fallback_rule,
                'sequence_length_bp': len(norm_seq),
                'qc_flags': qc_flags if qc_flags else 'none',
                'status': 'passed_qc'
            })
            
            # Collect sequence for FASTA output
            sequences_to_save.append({
                'header': f"{gene_name}|{gene_id}|{transcript_id}",
                'sequence': norm_seq,
                'gene_name': gene_name,
                'transcript_id': transcript_id,
                'length': len(norm_seq)
            })
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"  Processed {idx + 1} genes: {len(manifest_records)} passed, {len(failure_records)} failed")
        
        manifest_df = pd.DataFrame(manifest_records)
        failures_df = pd.DataFrame(failure_records)
        
        logger.info(f"Phase 2: Extraction complete:")
        logger.info(f"  Passed QC: {len(manifest_df)}")
        logger.info(f"  Failed: {len(failures_df)}")
        
        if error_counts:
            logger.info(f"  Error breakdown:")
            for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1])[:10]:
                logger.info(f"    {error_type}: {count}")
        
        # Combine stats
        stats = {
            **mapping_stats,
            'extracted': len(manifest_df),
            'extraction_failed': len(failures_df),
        }
        
        if len(failures_df) > 0:
            logger.info(f"  Failure breakdown:")
            for status, count in failures_df['status'].value_counts().items():
                logger.info(f"    {status}: {count}")
        
        # Extract unresolved genes from mapping
        unresolved_df = mapping_df[mapping_df['status'] != 'resolved'].copy() if 'status' in mapping_df.columns else pd.DataFrame()
        
        return manifest_df, failures_df, stats, unresolved_df, sequences_to_save
    
    def save_results(self, manifest_df: pd.DataFrame, failures_df: pd.DataFrame, stats: Dict, 
                     unresolved_df: pd.DataFrame = None, sequences: list = None):
        """Save manifest, failures, unresolved, sequences FASTA, and summary."""
        # Save manifest
        manifest_path = self.output_dir / 'manifest_mane.tsv'
        manifest_df.to_csv(manifest_path, sep='\t', index=False)
        logger.info(f"Saved manifest to {manifest_path}")
        
        # Save failures
        failures_path = self.output_dir / 'mane_extraction_failures.tsv'
        failures_df.to_csv(failures_path, sep='\t', index=False)
        logger.info(f"Saved failures to {failures_path}")
        
        # Save unresolved (always save, even if empty)
        unresolved_path = self.output_dir / 'unresolved_genes.tsv'
        if unresolved_df is not None:
            unresolved_df.to_csv(unresolved_path, sep='\t', index=False)
            logger.info(f"Saved unresolved genes to {unresolved_path} ({len(unresolved_df)} genes)")
        else:
            # Create empty file with headers if no unresolved
            pd.DataFrame(columns=['gene_name', 'gene_id', 'transcript_id', 'fallback_rule', 'status']).to_csv(
                unresolved_path, sep='\t', index=False)
            logger.info(f"Saved empty unresolved genes file to {unresolved_path}")
        
        # Save FASTA sequences
        if sequences and len(sequences) > 0:
            fasta_path = self.output_dir / 'mane_sequences.fasta'
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
            lengths_path = self.output_dir / 'mane_sequence_lengths.tsv'
            lengths_df = pd.DataFrame([
                {
                    'gene_name': entry['gene_name'],
                    'transcript_id': entry['transcript_id'],
                    'sequence_length_bp': entry['length']
                }
                for entry in sequences
            ])
            lengths_df.to_csv(lengths_path, sep='\t', index=False)
            logger.info(f"Saved sequence lengths to {lengths_path}")
        
        # Save summary
        summary_path = self.output_dir / 'mane_qc_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MANE TRANSCRIPT MANIFEST & QC SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write("[RESOLUTION PHASE]\n")
            f.write(f"  Input genes:              {stats['total_input']:,}\n")
            f.write(f"  Resolved:                 {stats['resolved']:,} ({100*stats['resolved']/stats['total_input']:.1f}%)\n")
            f.write(f"  Unresolved (not in GTF):  {stats['unresolved_not_in_gtf']:,}\n")
            f.write(f"  Unresolved (no txs):      {stats['unresolved_no_transcripts']:,}\n")
            
            f.write(f"\n[FALLBACK RULE USAGE]\n")
            f.write(f"  MANE Select:              {stats['used_mane_select']:,}\n")
            f.write(f"  Canonical tag:            {stats['used_canonical_tag']:,}\n")
            f.write(f"  APPRIS principal:         {stats['used_appris_principal']:,}\n")
            f.write(f"  Protein-coding:           {stats['used_protein_coding']:,}\n")
            f.write(f"  First available:          {stats['used_first_available']:,}\n")
            
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
    Build MANE transcript manifest from YAML config.
    
    Args:
        config_path: Path to YAML config file (e.g., config/config_mane.yaml)
        
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
    
    fasta_path = paths['fasta_path']
    gene_list_path = paths['gene_list_path']
    gtf_path = paths['annotation_path']
    output_dir = output_paths['reports_dir']
    max_length = embedding.get('max_length', 500000)
    
    builder = MANEManifestBuilder(
        fasta_path=fasta_path,
        gene_list_path=gene_list_path,
        gtf_path=gtf_path,
        output_dir=output_dir,
        max_length=max_length
    )
    
    manifest_df, failures_df, stats, unresolved_df, sequences = builder.process()
    builder.save_results(manifest_df, failures_df, stats, unresolved_df, sequences)
    
    logger.info("MANE manifest generation complete!")
    return manifest_df, failures_df, stats


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python manifest_mane.py <config.yaml>")
        print("\nExample:")
        print("  python manifest_mane.py ../../config/config_mane.yaml")
        sys.exit(1)
    
    main(config_path=sys.argv[1])
