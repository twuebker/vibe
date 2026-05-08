#!/usr/bin/env bash
# Download all ANN-Benchmarks datasets
# Source: https://github.com/erikbern/ann-benchmarks
#
# Most datasets are hosted on ann-benchmarks.com.
# The two COCO datasets are hosted on GitHub Releases (fabiocarrara/str-encoders).
#
# Usage:
#   ./download_ann_benchmarks.sh                  # download all datasets
#   ./download_ann_benchmarks.sh --dir /data/ann  # download to a specific directory

DEST_DIR="."

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

download() {
  local url="$1"
  local label="$2"
  echo ">>> Downloading $label ..."
  wget -c -P "$DEST_DIR" "$url"
}

echo "========================================"
echo "  ANN-Benchmarks Dataset Downloader"
echo "  Destination: $DEST_DIR"
echo "========================================"
echo ""

# --- Hosted on ann-benchmarks.com ---
ANN="http://ann-benchmarks.com"

download "$ANN/deep-image-96-angular.hdf5"          "DEEP1B (96d, Angular, 3.6GB)"
download "$ANN/fashion-mnist-784-euclidean.hdf5"    "Fashion-MNIST (784d, Euclidean, 217MB)"
download "$ANN/gist-960-euclidean.hdf5"             "GIST (960d, Euclidean, 3.6GB)"
download "$ANN/glove-100-angular.hdf5"              "GloVe-100 (Angular, 463MB)"
download "$ANN/mnist-784-euclidean.hdf5"            "MNIST (784d, Euclidean, 217MB)"
download "$ANN/nytimes-256-angular.hdf5"            "NYTimes (256d, Angular, 301MB)"
download "$ANN/sift-128-euclidean.hdf5"             "SIFT (128d, Euclidean, 501MB)"
download "$ANN/lastfm-64-dot.hdf5"                  "Last.fm (65d, Angular, 135MB)"

# --- Hosted on GitHub Releases (fabiocarrara/str-encoders) ---
COCO="https://github.com/fabiocarrara/str-encoders/releases/download/v0.1.3"

download "$COCO/coco-i2i-512-angular.hdf5"          "COCO-I2I (512d, Angular, 136MB)"
download "$COCO/coco-t2i-512-angular.hdf5"          "COCO-T2I (512d, Angular, 136MB)"

echo ""
echo "========================================"
echo "  All downloads complete!"
echo "========================================"
