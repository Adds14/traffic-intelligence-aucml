"""
PeMS04 Dataset Download Script
Downloads the PeMS04 traffic dataset from the ASTGCN repository.

Source: https://github.com/guoshnBJTU/ASTGCN-r-pytorch
Dataset: Caltrans Performance Measurement System, District 4 (San Francisco Bay Area)
Period: January-February 2018
Sensors: 307
Features: flow, occupancy, speed
Temporal Resolution: 5 minutes
"""

import os
import sys
import urllib.request
import zipfile
import shutil

# PeMS04 data URLs (from ASTGCN-r-pytorch repository)
DATA_URLS = {
    # Primary: Google Drive via gdown, or direct GitHub raw links
    # The standard PeMS04 release is distributed via several research repos
    "pems04_npz": "https://raw.githubusercontent.com/guoshnBJTU/ASTGCN-r-pytorch/master/data/PEMS04/pems04.npz",
    "distance_csv": "https://raw.githubusercontent.com/guoshnBJTU/ASTGCN-r-pytorch/master/data/PEMS04/distance.csv",
}

# Alternative URLs if primary fails
ALT_URLS = {
    "pems04_npz": [
        "https://github.com/guoshnBJTU/ASTGCN-r-pytorch/raw/master/data/PEMS04/pems04.npz",
    ],
    "distance_csv": [
        "https://github.com/guoshnBJTU/ASTGCN-r-pytorch/raw/master/data/PEMS04/distance.csv",
    ],
}


def download_file(url, dest_path, description="file"):
    """Download a file with progress reporting."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_path}")
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        file_size = os.path.getsize(dest_path)
        print(f"  Downloaded: {file_size / (1024*1024):.2f} MB")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def download_with_fallback(primary_url, alt_urls, dest_path, description):
    """Try primary URL, then fallback alternatives."""
    if download_file(primary_url, dest_path, description):
        return True
    
    for alt_url in alt_urls:
        print(f"  Trying alternative URL...")
        if download_file(alt_url, dest_path, description):
            return True
    
    return False


def main():
    # Determine project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from src/data/
    raw_dir = os.path.join(project_root, "data", "raw")
    
    os.makedirs(raw_dir, exist_ok=True)
    
    print("=" * 60)
    print("PeMS04 Dataset Download")
    print("=" * 60)
    
    # Check if files already exist
    npz_path = os.path.join(raw_dir, "pems04.npz")
    csv_path = os.path.join(raw_dir, "distance.csv")
    
    files_to_download = []
    
    if os.path.exists(npz_path):
        print(f"\npems04.npz already exists ({os.path.getsize(npz_path) / (1024*1024):.2f} MB)")
    else:
        files_to_download.append(("pems04_npz", npz_path, "PeMS04 traffic data (npz)"))
    
    if os.path.exists(csv_path):
        print(f"distance.csv already exists ({os.path.getsize(csv_path) / 1024:.2f} KB)")
    else:
        files_to_download.append(("distance_csv", csv_path, "PeMS04 distance matrix (csv)"))
    
    if not files_to_download:
        print("\nAll files already downloaded!")
        return True
    
    # Download missing files
    success = True
    for key, dest, desc in files_to_download:
        print()
        result = download_with_fallback(
            DATA_URLS[key],
            ALT_URLS.get(key, []),
            dest,
            desc
        )
        if not result:
            print(f"\n  ERROR: Failed to download {desc}")
            print(f"  Please manually download from one of the research repositories:")
            print(f"  - https://github.com/guoshnBJTU/ASTGCN-r-pytorch/tree/master/data/PEMS04")
            print(f"  - https://github.com/Davidham3/STSGCN/tree/master/data/PEMS04")
            print(f"  Place files in: {raw_dir}")
            success = False
    
    if success:
        print("\n" + "=" * 60)
        print("Download complete!")
        print(f"Files saved to: {raw_dir}")
        print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
