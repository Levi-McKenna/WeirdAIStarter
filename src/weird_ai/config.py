from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SAMPLE_LYRICS_FILE = PROCESSED_DATA_DIR / "lyrics_sample.txt"
TOKENS_FILE = PROCESSED_DATA_DIR / "tokens.txt"

WEIRD_AI_CFG = {
    "emb_dim": 768,
    "context_length": 1024,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
