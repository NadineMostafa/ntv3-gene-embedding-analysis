"""
Manifest and QC module for ntv3-gene-embedding-analysis.

Provides unified pipeline for building sequence manifests and performing
quality checks for both MANE and full genomic modes.

Both manifest builders follow a consistent structure:
  1. Parse annotation (GTF)
  2. Map input gene symbols to sequences (transcripts or genomic coordinates)
  3. Extract sequences from FASTA files
  4. Apply QC filters
  5. Generate manifest and failure reports

Field naming is consistent across both modes:
  - gene_name: input gene symbol
  - gene_id: Ensembl gene ID
  - status: 'resolved', 'unresolved_*', 'passed_qc', 'failed_*'
  
MANE-specific fields:
  - transcript_id: canonical transcript selected
  - sequence_length_nt: cDNA sequence length

Genomic-specific fields:
  - chromosome, start_bp, end_bp, strand: genomic coordinates
  - sequence_length_bp: genomic sequence length

Example Usage:
  # MANE mode
  from manifest_and_qc.manifest_mane import main as build_mane_manifest
  manifest_df, failures_df, stats = build_mane_manifest(
      fasta_path='data/reference/gencode.v45.transcripts.fa',
      gene_list_path='data/input/gene_list.txt',
      gtf_path='data/reference/gencode.v45.annotation.gtf',
      output_dir='data/outputs'
  )
  
  # Genomic mode
  from manifest_and_qc.manifest_genomic import main as build_genomic_manifest
  manifest_df, failures_df, stats = build_genomic_manifest(
      fasta_path='data/reference/GRCh38.primary_assembly.genome.fa',
      gene_list_path='data/input/gene_list.txt',
      gtf_path='data/reference/gencode.v45.annotation.gtf',
      output_dir='data/outputs'
  )
"""

__all__ = ['manifest_mane', 'manifest_genomic']
