# NTV3 Gene Embedding Analysis

Gene sequence embeddings using Nucleotide Transformer v3 (NTv3) with quality control and UMAP projection.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline (MANE mode - canonical transcripts)
python src/manifest_and_qc/manifest_mane.py config/config_mane.yaml
python src/inference/embed.py --config config/config_mane.yaml
python src/umap/Project_and_annotate.py --config config/config_mane.yaml
```

Outputs go to `data/outputs/MANE_sequence_650M/`

## Setup

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure JAX for GPU** (optional but recommended):
   ```bash
   pip install --upgrade jax jaxlib==0.4.28+cuda12_cudnn8.9 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

See "Reference Data Setup" section below for downloading GENCODE reference files.

## Configuration

Edit YAML configs in `config/`:
- **config_mane.yaml**: Uses canonical transcripts (faster, simpler)
- **config_genomic.yaml**: Uses full genomic loci (more context, slower)

Key parameters:
```yaml
model_name: NTv3_650M_pre      # or NTv3_100M_pre for smaller model
embedding:
  max_tokens_per_batch: 120_000    # Reduce if CUDA OOM errors
  max_sequence_length: 120_000
```

## Modes

| Mode | Sequence Type | Use Case |
|------|---------------|----------|
| MANE | Canonical transcript | Protein-coding genes |
| Genomic | Full locus | Regulatory regions, full context |

## Outputs

- **gene_embeddings.npy**: 1024-dim embeddings (650M) or 512-dim (100M)
- **phase6_umap_coordinates.tsv**: 2D UMAP projection coordinates
- **annotated_genes_metadata.tsv**: Gene annotations with cluster labels
- **reports/**: QC metrics and logs

## Repository Structure

```
ntv3-gene-embedding-analysis/
├── config/              # config_mane.yaml, config_genomic.yaml
├── data/
│   ├── input/          # Gene lists (gene_list.txt, etc)
│   ├── reference/      # GENCODE GTF, transcripts.fa, genome.fa
│   └── outputs/        # Results (MANE_sequence_650M/, FULL_sequence_650M/, etc)
└── src/
    ├── manifest_and_qc/     # Phase 1-2: sequence extraction
    ├── inference/           # Phase 3: embedding generation
    └── umap/                # Phase 4-5: projection & annotation
```

## Reference Data Setup

Download and extract GENCODE v45 reference files:

```bash
mkdir -p data/reference
cd data/reference

# Download all three reference files
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.transcripts.fa.gz
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/GRCh38.primary_assembly.genome.fa.gz

# Extract (remove .gz files after extraction)
gunzip gencode.v45.annotation.gtf.gz
gunzip gencode.v45.transcripts.fa.gz
gunzip GRCh38.primary_assembly.genome.fa.gz

cd ../..
```



## Config Files

Two YAML templates provided:
- **config_mane.yaml** — Canonical transcripts (faster)
- **config_genomic.yaml** — Full genomic loci (more context)

Edit paths, model name, batch size, and UMAP parameters in config.

## Running

MANE mode:
```bash
python src/manifest_and_qc/manifest_mane.py config/config_mane.yaml
python src/inference/embed.py --config config/config_mane.yaml
python src/umap/Project_and_annotate.py --config config/config_mane.yaml
```

Genomic mode: Replace `mane` with `genomic` in paths above.

## Output Files

- **gene_embeddings.npy** — Embedding matrix (N × 1536) for NTv3 650M and (N x 768) NTv3 for 100M 
- **phase6_umap_coordinates.tsv** — 2D coordinates with metadata
- **phase6_umap_scatter.png** — Visualization scatter plot
- **reports/** — QC metrics and extraction logs

## Citation

Nucleotide Transformer: Dalla-Torre et al. Nature Protocols (2023)  
GENCODE: Frankish et al. Nucleic Acids Research (2019)  
UMAP: McInnes, Healy & Melville, arXiv (2018)
