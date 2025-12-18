"""Download ARC-AGI datasets."""

import argparse
import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile
import shutil


def download_arc1(target_dir: Path):
    """
    Download ARC-AGI-1 dataset from GitHub.
    """
    print("Downloading ARC-AGI-1 dataset...")
    
    base_url = "https://api.github.com/repos/fchollet/ARC-AGI/contents/data"
    
    for split in ["training", "evaluation"]:
        split_dir = target_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Get file list
        response = requests.get(f"{base_url}/{split}")
        if response.status_code != 200:
            print(f"Error fetching {split} file list: {response.status_code}")
            continue
        
        files = response.json()
        
        for file_info in tqdm(files, desc=f"Downloading {split}"):
            if file_info["name"].endswith(".json"):
                # Download file
                download_url = file_info["download_url"]
                response = requests.get(download_url)
                
                if response.status_code == 200:
                    file_path = split_dir / file_info["name"]
                    with open(file_path, "w") as f:
                        json.dump(response.json(), f)
    
    print(f"ARC-AGI-1 downloaded to {target_dir}")


def download_arc2_from_kaggle(target_dir: Path):
    """
    Download ARC-AGI-2 from Kaggle.
    
    Requires kaggle CLI to be configured.
    """
    print("Downloading ARC-AGI-2 from Kaggle...")
    print("Note: Requires Kaggle API credentials")
    
    os.system(f"kaggle competitions download -c arc-prize-2025 -p {target_dir}")
    
    # Extract if downloaded as zip
    zip_path = target_dir / "arc-prize-2025.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        zip_path.unlink()
    
    print(f"ARC-AGI-2 downloaded to {target_dir}")


def create_sample_dataset(target_dir: Path, num_tasks: int = 10):
    """
    Create a small sample dataset for testing.
    
    This generates simple synthetic tasks for development.
    """
    print(f"Creating sample dataset with {num_tasks} tasks...")
    
    import numpy as np
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_tasks):
        task_id = f"sample_{i:04d}"
        
        # Generate a simple rotation task
        size = np.random.randint(3, 8)
        input_grid = np.random.randint(0, 5, (size, size)).tolist()
        output_grid = np.rot90(input_grid, k=-1).tolist()  # 90 degree rotation
        
        task = {
            "train": [
                {"input": input_grid, "output": output_grid}
                for _ in range(np.random.randint(2, 5))
            ],
            "test": [
                {
                    "input": np.random.randint(0, 5, (size, size)).tolist(),
                    "output": np.rot90(
                        np.random.randint(0, 5, (size, size)), k=-1
                    ).tolist()
                }
            ]
        }
        
        # Regenerate test with consistent transform
        for pair in task["test"]:
            pair["output"] = np.rot90(np.array(pair["input"]), k=-1).tolist()
        
        for pair in task["train"]:
            pair["output"] = np.rot90(np.array(pair["input"]), k=-1).tolist()
        
        with open(target_dir / f"{task_id}.json", "w") as f:
            json.dump(task, f, indent=2)
    
    print(f"Sample dataset created at {target_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download ARC-AGI datasets")
    parser.add_argument(
        "--version",
        choices=["arc1", "arc2", "both", "sample"],
        default="sample",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    if args.version == "arc1" or args.version == "both":
        download_arc1(output_dir / "arc-agi-1")
    
    if args.version == "arc2" or args.version == "both":
        download_arc2_from_kaggle(output_dir / "arc-agi-2")
    
    if args.version == "sample":
        create_sample_dataset(output_dir / "sample" / "training", num_tasks=20)
        create_sample_dataset(output_dir / "sample" / "evaluation", num_tasks=10)
    
    print("Done!")


if __name__ == "__main__":
    main()
