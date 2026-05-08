#!/usr/bin/env bash
# Download all VIBE (Vector Index Benchmark for Embeddings) datasets
# Source: https://github.com/vector-index-bench/vibe
#
# Usage:
#   ./download_vibe_datasets.sh                  # download all datasets
#   ./download_vibe_datasets.sh --dir /data/vibe # download to a specific directory
#   HF_TOKEN=hf_xxx ./download_vibe_datasets.sh  # with HuggingFace auth token (for gated datasets)

BASE_URL="https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main"
DEST_DIR="."

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      DEST_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 [--dir <destination_directory>]"
      exit 1
      ;;
  esac
done

mkdir -p "$DEST_DIR"

# Build wget options
WGET_OPTS=("-c" "-P" "$DEST_DIR")  # -c = resume partial downloads
if [[ -n "$HF_TOKEN" ]]; then
  WGET_OPTS+=("--header=Authorization: Bearer $HF_TOKEN")
fi

download() {
  local filename="$1"
  echo ">>> Downloading $filename ..."
  wget "${WGET_OPTS[@]}" "$BASE_URL/$filename"
}

echo "========================================"
echo "  VIBE Dataset Downloader"
echo "  Destination: $DEST_DIR"
echo "========================================"
echo ""

# --- In-distribution datasets ---
echo "--- In-distribution datasets ---"
download "agnews-mxbai-1024-euclidean.hdf5"
download "arxiv-nomic-768-normalized.hdf5"
download "ccnews-nomic-768-normalized.hdf5"
download "celeba-resnet-2048-cosine.hdf5"
download "codesearchnet-jina-768-cosine.hdf5"
download "glove-200-cosine.hdf5"
download "gooaq-distilroberta-768-normalized.hdf5"
download "imagenet-clip-512-normalized.hdf5"
download "landmark-dino-768-cosine.hdf5"
download "landmark-nomic-768-normalized.hdf5"
download "simplewiki-openai-3072-normalized.hdf5"
download "yahoo-minilm-384-normalized.hdf5"

# --- Out-of-distribution datasets ---
echo ""
echo "--- Out-of-distribution datasets ---"
download "coco-nomic-768-normalized.hdf5"
download "imagenet-align-640-normalized.hdf5"
download "laion-clip-512-normalized.hdf5"
download "yandex-200-cosine.hdf5"
download "yi-128-ip.hdf5"
download "llama-128-ip.hdf5"

echo ""
echo "========================================"
echo "  All downloads complete!"
echo "========================================"
