"""Build the FAISS index artifacts from the Kaggle CSVs.

    python scripts/build_index.py DATA_DIR [--out-dir artifacts/index] [--gcs-bucket BUCKET]
    python scripts/build_index.py --upload-only --gcs-bucket BUCKET [--out-dir artifacts/index]

DATA_DIR must contain movies_metadata.csv and credits.csv. Writes index.faiss, records.jsonl and
manifest.json, and exits with an error if the corpus size differs from --expected-count (41,328
for the full dataset). --upload-only skips the build: it checks the artifacts already in --out-dir
(file hashes, vector and record counts, embedding model) and uploads them unchanged.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from movie_rag import index as index_module
from movie_rag.config import get_settings
from movie_rag.preprocess import load_and_preprocess_movies, records_from_df
from movie_rag.retrieve import encode_documents, load_bi_encoder

FULL_CORPUS_SIZE = 41_328


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_hashes(out_dir: Path) -> dict[str, str]:
    return {name: sha256(out_dir / name) for name in (index_module.INDEX_FILE, index_module.RECORDS_FILE)}


def verify_artifacts(out_dir: Path, expected_count: int, embedding_model: str) -> dict:
    """Reload the artifacts in out_dir and check them against each other and the manifest."""
    movie_index = index_module.load(out_dir, expected_embedding_model=embedding_model)
    if movie_index.size != expected_count:
        raise index_module.IndexArtifactError(f"expected {expected_count:,} documents, found {movie_index.size:,}")
    if movie_index.manifest.get("sha256") != file_hashes(out_dir):
        raise index_module.IndexArtifactError(
            f"the files in {out_dir} do not match the sha256 hashes in {index_module.MANIFEST_FILE}"
        )
    return movie_index.manifest


def upload_to_gcs(out_dir: Path, bucket_name: str, prefix: str, project: str | None) -> None:
    try:
        from google.cloud import storage  # preinstalled on Colab
    except ImportError:
        sys.exit('error: google-cloud-storage is not installed; run pip install -e ".[gcs]"')

    bucket = storage.Client(project=project).bucket(bucket_name)
    for name in (index_module.INDEX_FILE, index_module.RECORDS_FILE, index_module.MANIFEST_FILE):
        blob = bucket.blob(f"{prefix.strip('/')}/{name}")
        blob.upload_from_filename(str(out_dir / name))
        print(f"Uploaded gs://{bucket_name}/{blob.name}")


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, nargs="?", help="folder with movies_metadata.csv and credits.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/index"))
    parser.add_argument("--expected-count", type=int, default=FULL_CORPUS_SIZE)
    parser.add_argument("--batch-size", type=int, default=settings.embed_batch_size)
    parser.add_argument("--device", default=settings.device, help="auto, cpu, cuda or mps")
    parser.add_argument("--gcs-bucket", help="also upload the artifacts to this private GCS bucket")
    parser.add_argument("--gcs-prefix", default="index")
    parser.add_argument("--gcs-project", help="GCP project for the storage client (needed on Colab)")
    parser.add_argument(
        "--upload-only", action="store_true", help="skip the build; check and upload the artifacts in --out-dir"
    )
    args = parser.parse_args()

    if args.upload_only:
        if args.data_dir or not args.gcs_bucket:
            parser.error("--upload-only takes --gcs-bucket and no DATA_DIR")
        try:
            manifest = verify_artifacts(args.out_dir, args.expected_count, settings.embedding_model)
        except index_module.IndexArtifactError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Checked {manifest['doc_count']:,} documents in {args.out_dir} (built {manifest.get('created_at')})")
        upload_to_gcs(args.out_dir, args.gcs_bucket, args.gcs_prefix, args.gcs_project)
        return 0
    if args.data_dir is None:
        parser.error("DATA_DIR is required unless --upload-only is given")

    movies_csv, credits_csv = args.data_dir / "movies_metadata.csv", args.data_dir / "credits.csv"
    for path in (movies_csv, credits_csv):
        if not path.exists():
            print(f"error: {path} not found", file=sys.stderr)
            return 2

    df, stats = load_and_preprocess_movies(movies_csv, credits_csv)
    print(f"Raw rows: {stats['raw_movie_rows']:,}")
    print(f"Document count: {len(df):,}")
    if len(df) != args.expected_count:
        print(f"error: expected {args.expected_count:,} documents, got {len(df):,}", file=sys.stderr)
        return 1

    device = settings.model_copy(update={"device": args.device}).resolved_device()
    encoder = load_bi_encoder(settings.embedding_model, device)
    start = time.perf_counter()
    embeddings = encode_documents(encoder, df["document_text"].tolist(), args.batch_size, show_progress=True)
    encode_s = time.perf_counter() - start
    print(f"Encoded {len(df):,} documents in {encode_s:.1f}s on {device} ({len(df) / encode_s:.1f} docs/sec)")

    faiss_index = index_module.build_faiss_index(embeddings)
    manifest = {
        "doc_count": int(faiss_index.ntotal),
        "dim": int(embeddings.shape[1]),
        "embedding_model": settings.embedding_model,
        "cross_encoder_model": settings.cross_encoder_model,
        "faiss_index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "encode_device": device,
        "encode_seconds": round(encode_s, 2),
        "corpus_stats": stats,
    }
    index_module.save(args.out_dir, faiss_index, records_from_df(df), manifest)
    manifest["sha256"] = file_hashes(args.out_dir)
    (args.out_dir / index_module.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote the index artifacts to {args.out_dir}")

    if args.gcs_bucket:
        upload_to_gcs(args.out_dir, args.gcs_bucket, args.gcs_prefix, args.gcs_project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
