"""Download the bi-encoder and cross-encoder into the Hugging Face cache (used at Docker build time)."""

from movie_rag.config import get_settings
from movie_rag.retrieve import load_bi_encoder, load_cross_encoder

if __name__ == "__main__":
    settings = get_settings()
    load_bi_encoder(settings.embedding_model, "cpu")
    load_cross_encoder(settings.cross_encoder_model, "cpu")
    print(f"Cached {settings.embedding_model} and {settings.cross_encoder_model}")
