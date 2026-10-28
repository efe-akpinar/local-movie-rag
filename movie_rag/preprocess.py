"""Corpus preprocessing, copied from the notebook's data pipeline (Section 2)."""

import ast
from pathlib import Path

import pandas as pd

CUTOFF_DATE = pd.Timestamp("2017-07-31")

RECORD_COLUMNS = [
    "row_id",
    "movie_id",
    "title",
    "release_year",
    "genres",
    "director",
    "cast_names",
    "overview",
    "document_text",
]


def parse_jsonish_list(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def extract_names(value, limit=None):
    items = parse_jsonish_list(value)
    names = []
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
    if limit is not None:
        names = names[:limit]
    return names


def extract_directors(value):
    items = parse_jsonish_list(value)
    directors = []
    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director":
            name = str(item.get("name", "")).strip()
            if name:
                directors.append(name)
    return ", ".join(directors) if directors else "Unknown"


def list_to_text(values, empty="Unknown"):
    if isinstance(values, list) and values:
        return ", ".join(str(v) for v in values if str(v).strip())
    if isinstance(values, str) and values.strip():
        return values
    return empty


def build_document_text(row):
    year = "Unknown" if pd.isna(row["release_year"]) else str(int(row["release_year"]))
    genres = list_to_text(row["genres"], empty="Unknown")
    cast = list_to_text(row["cast_names"], empty="Unknown")
    director = row["director"] if isinstance(row["director"], str) and row["director"].strip() else "Unknown"
    return (
        f"Title: {row['title']}\n"
        f"Year: {year}\n"
        f"Genres: {genres}\n"
        f"Director: {director}\n"
        f"Cast: {cast}\n"
        f"Overview: {row['overview']}"
    )


def load_and_preprocess_movies(
    movies_csv: Path, credits_csv: Path, cutoff_date: pd.Timestamp = CUTOFF_DATE
) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned corpus (one row per movie) and summary statistics."""
    movies = pd.read_csv(movies_csv, low_memory=False)
    credits = pd.read_csv(credits_csv)

    raw_movie_rows = len(movies)

    movies = movies.dropna(subset=["title"]).copy()
    movies["overview"] = movies["overview"].fillna("").astype(str)
    movies = movies[movies["overview"].str.strip().ne("")].copy()

    movies["movie_id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["movie_id"]).copy()
    movies["movie_id"] = movies["movie_id"].astype("int64")

    movies["release_dt"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies = movies[(movies["release_dt"].isna()) | (movies["release_dt"] <= cutoff_date)].copy()
    movies["release_year"] = movies["release_dt"].dt.year.astype("Int64")

    movies["genres"] = movies["genres"].apply(lambda value: extract_names(value, limit=None))
    movies["overview_len_tokens"] = movies["overview"].str.split().str.len()

    movies = (
        movies.sort_values(["title", "overview_len_tokens"], ascending=[True, False])
        .drop_duplicates(subset=["title"], keep="first")
        .copy()
    )

    credits["movie_id"] = pd.to_numeric(credits["id"], errors="coerce")
    credits = credits.dropna(subset=["movie_id"]).copy()
    credits["movie_id"] = credits["movie_id"].astype("int64")
    credits["director"] = credits["crew"].apply(extract_directors)
    credits["cast_names"] = credits["cast"].apply(lambda value: extract_names(value, limit=5))
    credits = credits[["movie_id", "director", "cast_names"]].drop_duplicates(subset=["movie_id"])

    df = movies.merge(credits, on="movie_id", how="left")
    df["director"] = df["director"].fillna("Unknown")
    df["cast_names"] = df["cast_names"].apply(lambda value: value if isinstance(value, list) else [])
    df["document_text"] = df.apply(build_document_text, axis=1)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index.astype(int)

    df = df[RECORD_COLUMNS[:-1] + ["overview_len_tokens", "document_text"]].copy()

    stats = {
        "raw_movie_rows": int(raw_movie_rows),
        "corpus_size": int(len(df)),
        "overview_tokens_mean": float(df["overview_len_tokens"].mean()),
        "overview_tokens_median": float(df["overview_len_tokens"].median()),
        "overview_tokens_max": int(df["overview_len_tokens"].max()),
    }
    return df, stats


def records_from_df(df: pd.DataFrame) -> list[dict]:
    clean_df = df[RECORD_COLUMNS].copy()
    clean_df["release_year"] = clean_df["release_year"].astype("object")
    clean_df = clean_df.where(pd.notnull(clean_df), None)
    return clean_df.to_dict("records")
