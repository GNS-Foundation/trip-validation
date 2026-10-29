#!/usr/bin/env bash
# Fetch the datasets the lab runs on. These are NOT committed (see .gitignore).
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p data geo

echo "Fetching GeoLife GPS Trajectories 1.3 (~313 MB, Microsoft Research)..."
curl -L -o data/geolife.zip \
  "https://download.microsoft.com/download/F/4/8/F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip"
( cd data && unzip -q -o geolife.zip )
echo "  -> data/Geolife Trajectories 1.3/Data/<user>/Trajectory/*.plt"

echo "Fetching Netherlands boundary polygon (for the geographic worked example)..."
curl -L -o geo/netherlands.json \
  "https://raw.githubusercontent.com/georgique/world-geojson/develop/countries/netherlands.json"
echo "  -> geo/netherlands.json"

echo "Done."
