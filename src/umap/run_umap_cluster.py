"""
Phase 6: UMAP projection for gene embeddings.

Configuration-driven UMAP projection and annotation for MANE embeddings.
"""

import argparse
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_optional_module(module_name: str, install_hint: str):
    """Import an optional dependency with a clear installation hint."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(install_hint) from exc


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration for the MANE pipeline."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file did not parse into a dictionary: {config_path}")

    return config


def resolve_paths(config_dir: Path, config: Dict) -> Dict:
    """Resolve relative config paths against the config file directory."""
    resolved_config = dict(config)
    paths = dict(resolved_config.get("paths", {}))

    for section_name in ("input", "output"):
        section = dict(paths.get(section_name, {}))
        for key, value in section.items():
            if isinstance(value, str) and not Path(value).is_absolute():
                section[key] = str((config_dir / value).resolve())
        paths[section_name] = section

    umap_config = dict(resolved_config.get("umap", {}))
    annotation_sources = dict(umap_config.get("annotation_sources", {}))
    for key, value in annotation_sources.items():
        if isinstance(value, str) and not Path(value).is_absolute():
            annotation_sources[key] = str((config_dir / value).resolve())
    umap_config["annotation_sources"] = annotation_sources

    output_dir = umap_config.get("output_dir")
    if isinstance(output_dir, str) and not Path(output_dir).is_absolute():
        umap_config["output_dir"] = str((config_dir / output_dir).resolve())

    resolved_config["paths"] = paths
    resolved_config["umap"] = umap_config
    return resolved_config


def load_transcript_ids(ids_path: Path) -> List[str]:
    """Load transcript IDs aligned to embedding rows."""
    with ids_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def load_requested_genes(gene_path: Path) -> List[str]:
    """Load requested genes from either a plain-text or simple tabular file."""
    with gene_path.open("r", encoding="utf-8") as handle:
        non_empty_lines = [line.strip() for line in handle if line.strip()]

    if not non_empty_lines:
        return []

    first_line = non_empty_lines[0]
    if "\t" not in first_line and "," not in first_line:
        return [line.strip() for line in non_empty_lines if line.strip()]

    gene_frame = pd.read_csv(gene_path, sep=None, engine="python")
    if gene_frame.empty:
        return []

    first_column = gene_frame.columns[0]
    return [
        str(value).strip()
        for value in gene_frame[first_column].dropna().tolist()
        if str(value).strip()
    ]


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load manifest and keep the columns needed for traceability."""
    manifest = pd.read_csv(manifest_path, sep="\t")

    required = {"transcript_id", "gene_name", "gene_id"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    keep_columns = [
        column
        for column in [
            "transcript_id",
            "gene_id",
            "gene_name",
            "fallback_rule",
            "sequence_length_nt",
            "qc_flags",
            "status",
        ]
        if column in manifest.columns
    ]

    return manifest[keep_columns].drop_duplicates(subset=["transcript_id"], keep="first")


def resolve_annotation_rows(
    requested_genes: List[str],
    manifest: pd.DataFrame,
    transcript_ids: List[str],
) -> pd.DataFrame:
    """
    Resolve requested genes to exactly one embedding row each.

    Assumes the MANE manifest contains one transcript per gene embedding.
    """
    ids_to_index = {transcript_id: idx for idx, transcript_id in enumerate(transcript_ids)}

    manifest_lookup = manifest.copy()
    manifest_lookup["embedding_row_index"] = manifest_lookup["transcript_id"].map(ids_to_index)

    manifest_by_gene_name = manifest_lookup.drop_duplicates(subset=["gene_name"], keep="first")
    manifest_by_gene_id = manifest_lookup.drop_duplicates(subset=["gene_id"], keep="first")
    manifest_by_transcript = manifest_lookup.drop_duplicates(subset=["transcript_id"], keep="first")

    records = []
    for requested_gene in requested_genes:
        match = manifest_by_gene_name[manifest_by_gene_name["gene_name"] == requested_gene]
        match_type = "gene_name"

        if match.empty:
            match = manifest_by_gene_id[manifest_by_gene_id["gene_id"] == requested_gene]
            match_type = "gene_id"
        if match.empty:
            match = manifest_by_transcript[manifest_by_transcript["transcript_id"] == requested_gene]
            match_type = "transcript_id"

        if match.empty:
            records.append(
                {
                    "requested_gene": requested_gene,
                    "match_type": None,
                    "gene_name": None,
                    "gene_id": None,
                    "transcript_id": None,
                    "embedding_row_index": None,
                    "found": False,
                }
            )
            continue

        row = match.iloc[0]
        records.append(
            {
                "requested_gene": requested_gene,
                "match_type": match_type,
                "gene_name": row["gene_name"],
                "gene_id": row["gene_id"],
                "transcript_id": row["transcript_id"],
                "embedding_row_index": int(row["embedding_row_index"]),
                "found": pd.notna(row["embedding_row_index"]),
            }
        )

    annotation_table = pd.DataFrame(records)
    if not annotation_table.empty:
        annotation_table["found"] = annotation_table["found"].fillna(False).astype(bool)

    return annotation_table


def build_annotation_membership(
    annotation_sources: Dict[str, Path],
    manifest: pd.DataFrame,
    transcript_ids: List[str],
    total_embeddings: int,
    strict: bool,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, List[str]]]:
    """Build per-row membership and a concrete annotation mapping table."""
    membership = np.empty(total_embeddings, dtype=object)
    membership[:] = ""
    mapping_frames: List[pd.DataFrame] = []
    missing_by_source: Dict[str, List[str]] = {}

    for source_name, gene_path in annotation_sources.items():
        requested_genes = load_requested_genes(gene_path)
        annotation_rows = resolve_annotation_rows(
            requested_genes=requested_genes,
            manifest=manifest,
            transcript_ids=transcript_ids,
        )
        annotation_rows.insert(0, "annotation_source", source_name)
        mapping_frames.append(annotation_rows)

        found_rows = annotation_rows[annotation_rows["found"]]
        for row_index in found_rows["embedding_row_index"].astype(int).tolist():
            if membership[row_index]:
                existing = set(membership[row_index].split(";"))
                existing.add(source_name)
                membership[row_index] = ";".join(sorted(existing))
            else:
                membership[row_index] = source_name

        missing_by_source[source_name] = (
            annotation_rows.loc[~annotation_rows["found"], "requested_gene"].astype(str).tolist()
        )
        if strict and missing_by_source[source_name]:
            raise ValueError(
                f"Annotation source '{source_name}' contains unresolved genes: {missing_by_source[source_name]}"
            )

    mapping_table = pd.concat(mapping_frames, ignore_index=True) if mapping_frames else pd.DataFrame()
    return membership, mapping_table, missing_by_source


def build_projection_table(
    transcript_ids: List[str],
    coordinates: np.ndarray,
    manifest: Optional[pd.DataFrame],
    annotation_sources: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Build a row-aligned projection table with optional gene metadata and annotations."""
    table = pd.DataFrame(
        {
            "row_index": np.arange(len(transcript_ids), dtype=int),
            "transcript_id": transcript_ids,
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
        }
    )

    if annotation_sources is not None:
        table["annotation_sources"] = annotation_sources
        table["is_annotated"] = table["annotation_sources"].astype(str) != ""

    if manifest is not None:
        table = table.merge(manifest, on="transcript_id", how="left", validate="many_to_one")
        unresolved = int(table["gene_name"].isna().sum()) if "gene_name" in table.columns else 0
        if unresolved > 0:
            logger.warning(
                "%d projected rows could not be matched to manifest metadata by transcript_id",
                unresolved,
            )

    return table


def run_umap_projection(
    embeddings: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_seed: int,
    n_components: int,
) -> np.ndarray:
    """Fit UMAP and return coordinates."""
    umap_module = load_optional_module(
        module_name="umap",
        install_hint="UMAP dependency not found. Install with: pip install umap-learn",
    )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-8)

    reducer = umap_module.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_seed,
    )
    return reducer.fit_transform(embeddings_normalized)


def save_plot(
    plot_path: Path,
    coordinates: np.ndarray,
    title: str,
    annotated_mask: Optional[np.ndarray] = None,
    annotated_color: str = "red",
    default_color: str = "lightgray",
) -> None:
    """Save a static UMAP scatter plot with optional annotation highlighting."""
    matplotlib = load_optional_module(
        module_name="matplotlib",
        install_hint="Matplotlib dependency not found. Install with: pip install matplotlib",
    )
    matplotlib.use("Agg")
    plt = load_optional_module(
        module_name="matplotlib.pyplot",
        install_hint="Matplotlib dependency not found. Install with: pip install matplotlib",
    )

    plt.figure(figsize=(12, 9), dpi=150)

    if annotated_mask is not None:
        non_annotated = ~annotated_mask
        if np.any(non_annotated):
            plt.scatter(
                coordinates[non_annotated, 0],
                coordinates[non_annotated, 1],
                c=default_color,
                s=5,
                alpha=0.4,
                linewidths=0,
                label="Other transcripts",
            )

        if np.any(annotated_mask):
            plt.scatter(
                coordinates[annotated_mask, 0],
                coordinates[annotated_mask, 1],
                c=annotated_color,
                s=25,
                alpha=0.85,
                linewidths=0.5,
                edgecolors="darkred" if annotated_color == "red" else "black",
                label="Annotated transcripts",
            )
        plt.legend(loc="best", framealpha=0.9)
    else:
        plt.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            s=3,
            alpha=0.65,
            linewidths=0,
        )

    plt.title(title, fontsize=14, fontweight="bold" if annotated_mask is not None else "normal")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()


def save_summary(
    summary_path: Path,
    embeddings: np.ndarray,
    coordinates: np.ndarray,
    projection_table: pd.DataFrame,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_seed: int,
    n_components: int,
    annotated_rows: int,
    annotation_counts: Dict[str, int],
) -> None:
    """Save run metadata and basic QC summary."""
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("Phase 6 UMAP Summary\n")
        handle.write("=" * 50 + "\n")
        handle.write(f"Input embedding shape: {embeddings.shape}\n")
        handle.write(f"Output coordinate shape: {coordinates.shape}\n")
        handle.write(f"Rows in projection table: {len(projection_table)}\n")
        handle.write(f"Annotated rows in visualization: {annotated_rows}\n")
        handle.write("\n")
        handle.write("UMAP parameters\n")
        handle.write("-" * 50 + "\n")
        handle.write(f"n_components: {n_components}\n")
        handle.write(f"n_neighbors: {n_neighbors}\n")
        handle.write(f"min_dist: {min_dist}\n")
        handle.write(f"metric: {metric}\n")
        handle.write(f"random_seed: {random_seed}\n")
        handle.write("\n")
        handle.write("Annotation counts\n")
        handle.write("-" * 50 + "\n")
        if annotation_counts:
            for source_name, count in annotation_counts.items():
                handle.write(f"{source_name}: {count}\n")
        else:
            handle.write("No annotation sources configured\n")
        handle.write("\n")
        handle.write("Coordinate range\n")
        handle.write("-" * 50 + "\n")
        handle.write(f"UMAP-1 min/max: {coordinates[:, 0].min():.6f} / {coordinates[:, 0].max():.6f}\n")
        handle.write(f"UMAP-2 min/max: {coordinates[:, 1].min():.6f} / {coordinates[:, 1].max():.6f}\n")
        handle.write("\n")
        handle.write("Numeric checks\n")
        handle.write("-" * 50 + "\n")
        handle.write(f"Embedding contains NaN: {bool(np.isnan(embeddings).any())}\n")
        handle.write(f"Embedding contains Inf: {bool(np.isinf(embeddings).any())}\n")
        handle.write(f"Coordinates contain NaN: {bool(np.isnan(coordinates).any())}\n")
        handle.write(f"Coordinates contain Inf: {bool(np.isinf(coordinates).any())}\n")


def decode_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Decode embeddings to float32, including raw bfloat16 storage."""
    if embeddings.dtype.kind == "V":
        if not HAS_JAX:
            raise ImportError("JAX required to decode bfloat16 embeddings. Install with: pip install jax")
        return np.asarray(embeddings.view(jnp.bfloat16), dtype=np.float32)
    return np.asarray(embeddings, dtype=np.float32)


def get_annotation_configuration(umap_config: Dict) -> Tuple[Dict[str, Path], str, str, bool]:
    """Extract annotation configuration from the UMAP config section."""
    annotation_sources_raw = umap_config.get("annotation_sources", {})
    if not isinstance(annotation_sources_raw, dict):
        raise ValueError("umap.annotation_sources must be a mapping of label -> gene list path")

    annotation_sources = {
        source_name: Path(source_path)
        for source_name, source_path in annotation_sources_raw.items()
        if source_path
    }

    for source_name, source_path in annotation_sources.items():
        if not source_path.exists():
            raise FileNotFoundError(
                f"Configured annotation source '{source_name}' not found: {source_path}"
            )

    annotated_color = str(umap_config.get("annotated_color", "red"))
    default_color = str(umap_config.get("default_color", "lightgray"))
    strict = bool(umap_config.get("annotation_strict", False))

    return annotation_sources, annotated_color, default_color, strict


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: Project embeddings onto UMAP")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config_mane.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config = resolve_paths(args.config.parent.resolve(), load_config(args.config))
    output_paths = config.get("paths", {}).get("output", {})
    umap_config = config.get("umap", {})

    embeddings_path = Path(output_paths["embeddings_path"])
    ids_path = Path(output_paths["ids_path"])
    manifest_path = Path(output_paths["manifest_path"])
    output_dir = Path(umap_config["output_dir"])

    n_neighbors = int(umap_config.get("n_neighbors", 15))
    min_dist = float(umap_config.get("min_dist", 0.0))
    n_components = int(umap_config.get("n_components", 2))
    metric = str(umap_config.get("metric", "cosine"))
    seed = int(umap_config.get("seed", 42))

    if n_neighbors < 2:
        raise ValueError("umap.n_neighbors must be >= 2")
    if n_components != 2:
        raise ValueError("run_umap_cluster.py currently expects umap.n_components to be 2")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embeddings from %s", embeddings_path)
    embeddings = decode_embeddings(np.load(embeddings_path, allow_pickle=True))

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape: {embeddings.shape}")
    if np.isnan(embeddings).any() or np.isinf(embeddings).any():
        raise ValueError("Embedding matrix contains NaN/Inf values; fix embedding output before UMAP")

    transcript_ids = load_transcript_ids(ids_path)
    if len(transcript_ids) != embeddings.shape[0]:
        raise ValueError(
            "ID row mismatch: "
            f"{len(transcript_ids)} IDs vs {embeddings.shape[0]} embedding rows"
        )

    logger.info("Loading manifest from %s", manifest_path)
    manifest = load_manifest(manifest_path)

    annotation_sources, annotated_color, default_color, strict = get_annotation_configuration(umap_config)

    membership = None
    annotated_mask = None
    annotation_mapping = pd.DataFrame()
    annotation_counts: Dict[str, int] = {}
    missing_by_source: Dict[str, List[str]] = {}

    if annotation_sources:
        membership, annotation_mapping, missing_by_source = build_annotation_membership(
            annotation_sources=annotation_sources,
            manifest=manifest,
            transcript_ids=transcript_ids,
            total_embeddings=embeddings.shape[0],
            strict=strict,
        )
        annotated_mask = membership != ""
        for source_name in annotation_sources:
            annotation_counts[source_name] = int(np.sum([source_name in label.split(";") if label else False for label in membership]))
            if missing_by_source.get(source_name):
                logger.warning(
                    "Annotation source '%s' had %d genes with no manifest/embedding match",
                    source_name,
                    len(missing_by_source[source_name]),
                )
    else:
        logger.info("No annotation sources configured; generating unannotated UMAP output")

    logger.info(
        "Running UMAP (n_neighbors=%d, min_dist=%s, metric=%s, seed=%d)",
        n_neighbors,
        min_dist,
        metric,
        seed,
    )
    coordinates = run_umap_projection(
        embeddings=embeddings,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_seed=seed,
        n_components=n_components,
    )

    projection_table = build_projection_table(
        transcript_ids=transcript_ids,
        coordinates=coordinates,
        manifest=manifest,
        annotation_sources=membership,
    )

    coord_npy_path = output_dir / "phase6_umap_coordinates.npy"
    coord_tsv_path = output_dir / "phase6_umap_coordinates.tsv"
    plot_path = output_dir / "phase6_umap_scatter.png"
    summary_path = output_dir / "phase6_umap_summary.txt"
    metadata_path = output_dir / "annotated_genes_metadata.tsv"
    mapping_path = output_dir / "annotated_gene_row_mapping.tsv"
    missing_path = output_dir / "annotation_missing_genes.tsv"

    np.save(coord_npy_path, coordinates)
    projection_table.to_csv(coord_tsv_path, sep="\t", index=False)

    title = "Gene Embeddings UMAP Projection"
    if annotation_counts:
        title += f"\n(Annotated rows: {int(np.sum(annotated_mask))})"

    save_plot(
        plot_path=plot_path,
        coordinates=coordinates,
        title=title,
        annotated_mask=annotated_mask,
        annotated_color=annotated_color,
        default_color=default_color,
    )

    save_summary(
        summary_path=summary_path,
        embeddings=embeddings,
        coordinates=coordinates,
        projection_table=projection_table,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_seed=seed,
        n_components=n_components,
        annotated_rows=int(np.sum(annotated_mask)) if annotated_mask is not None else 0,
        annotation_counts=annotation_counts,
    )

    if not annotation_mapping.empty:
        annotation_mapping.to_csv(mapping_path, sep="\t", index=False)

        found_mappings = annotation_mapping[annotation_mapping["found"]].copy()
        if not found_mappings.empty:
            metadata_columns = [
                column
                for column in [
                    "annotation_source",
                    "requested_gene",
                    "gene_name",
                    "gene_id",
                    "transcript_id",
                ]
                if column in found_mappings.columns
            ]
            found_mappings[metadata_columns].drop_duplicates().to_csv(metadata_path, sep="\t", index=False)

    if annotation_sources:
        missing_records = [
            {"annotation_source": source_name, "gene": gene}
            for source_name, genes in missing_by_source.items()
            for gene in genes
        ]
        pd.DataFrame(missing_records, columns=["annotation_source", "gene"]).to_csv(
            missing_path,
            sep="\t",
            index=False,
        )

    logger.info("Saved UMAP coordinates to %s", coord_npy_path)
    logger.info("Saved projection table to %s", coord_tsv_path)
    logger.info("Saved scatter plot to %s", plot_path)
    logger.info("Saved run summary to %s", summary_path)
    if not annotation_mapping.empty:
        logger.info("Saved annotation row mapping to %s", mapping_path)
    if not annotation_mapping.empty:
        logger.info("Saved annotated genes metadata to %s", metadata_path)
    if annotation_sources:
        logger.info("Saved missing annotation report to %s", missing_path)


if __name__ == "__main__":
    main()
