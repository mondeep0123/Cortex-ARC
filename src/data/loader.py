"""Data loader for ARC-AGI datasets."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Union
from dataclasses import dataclass
import requests
from tqdm import tqdm

from ..core.task import Task


# Official ARC dataset URLs
ARC_URLS = {
    "arc1": {
        "training": "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/training",
        "evaluation": "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data/evaluation",
    },
    "arc2": {
        "training": "https://raw.githubusercontent.com/arc-community/arc-agi-2/main/data/training",
        "evaluation": "https://raw.githubusercontent.com/arc-community/arc-agi-2/main/data/evaluation",
    }
}


@dataclass
class ARCDataset:
    """
    Dataset class for ARC-AGI tasks.
    
    Handles loading, caching, and iterating over ARC tasks.
    """
    
    tasks: Dict[str, Task]
    version: str  # "arc1" or "arc2"
    split: str    # "training", "evaluation", or "test"
    
    def __len__(self) -> int:
        return len(self.tasks)
    
    def __getitem__(self, task_id: str) -> Task:
        return self.tasks[task_id]
    
    def __iter__(self) -> Iterator[Task]:
        return iter(self.tasks.values())
    
    def __contains__(self, task_id: str) -> bool:
        return task_id in self.tasks
    
    def get(self, task_id: str) -> Optional[Task]:
        """Get a task by ID, returns None if not found."""
        return self.tasks.get(task_id)
    
    def task_ids(self) -> List[str]:
        """Get all task IDs."""
        return list(self.tasks.keys())
    
    def sample(self, n: int = 1) -> List[Task]:
        """Randomly sample n tasks."""
        import random
        ids = random.sample(self.task_ids(), min(n, len(self)))
        return [self.tasks[tid] for tid in ids]
    
    def filter(self, predicate) -> ARCDataset:
        """Filter tasks by a predicate function."""
        filtered = {tid: task for tid, task in self.tasks.items() if predicate(task)}
        return ARCDataset(tasks=filtered, version=self.version, split=self.split)
    
    def summary(self) -> str:
        """Generate a summary of the dataset."""
        total_train_pairs = sum(t.num_train for t in self.tasks.values())
        total_test_pairs = sum(t.num_test for t in self.tasks.values())
        
        return f"""ARC Dataset Summary:
  Version: {self.version}
  Split: {self.split}
  Total tasks: {len(self)}
  Total training pairs: {total_train_pairs}
  Total test pairs: {total_test_pairs}
  Avg training pairs per task: {total_train_pairs / len(self):.1f}
  Avg test pairs per task: {total_test_pairs / len(self):.1f}
"""
    
    def __repr__(self) -> str:
        return f"ARCDataset(version={self.version}, split={self.split}, tasks={len(self)})"


def load_from_directory(directory: Path, version: str = "arc1", split: str = "training") -> ARCDataset:
    """
    Load ARC tasks from a directory of JSON files.
    
    Args:
        directory: Path to directory containing JSON task files
        version: "arc1" or "arc2"
        split: "training", "evaluation", or "test"
    
    Returns:
        ARCDataset containing all loaded tasks
    """
    tasks = {}
    
    json_files = list(directory.glob("*.json"))
    
    for json_file in tqdm(json_files, desc=f"Loading {split}"):
        try:
            task = Task.from_json(json_file)
            tasks[task.task_id] = task
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    return ARCDataset(tasks=tasks, version=version, split=split)


def download_dataset(
    version: str = "arc1",
    split: str = "training", 
    target_dir: Optional[Path] = None
) -> Path:
    """
    Download ARC dataset from GitHub.
    
    Args:
        version: "arc1" or "arc2"
        split: "training" or "evaluation"
        target_dir: Where to save files (default: data/{version}/{split})
    
    Returns:
        Path to the downloaded directory
    """
    if target_dir is None:
        target_dir = Path("data") / version / split
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # This is a placeholder - in practice you'd need to:
    # 1. Get the list of files from the GitHub API
    # 2. Download each file
    
    print(f"Dataset should be downloaded to: {target_dir}")
    print(f"Please clone the ARC repository manually:")
    print(f"  git clone https://github.com/fchollet/ARC-AGI.git")
    print(f"Or download from Kaggle for ARC-AGI-2")
    
    return target_dir


def load_arc1(
    data_dir: Union[str, Path] = "data/arc-agi-1",
    splits: List[str] = ["training", "evaluation"]
) -> Dict[str, ARCDataset]:
    """
    Load ARC-AGI-1 dataset.
    
    Args:
        data_dir: Root directory of ARC-AGI-1 data
        splits: Which splits to load
    
    Returns:
        Dictionary mapping split name to ARCDataset
    """
    data_dir = Path(data_dir)
    datasets = {}
    
    for split in splits:
        split_dir = data_dir / split
        if split_dir.exists():
            datasets[split] = load_from_directory(split_dir, version="arc1", split=split)
        else:
            print(f"Warning: {split_dir} not found. Run scripts/download_data.py first.")
    
    return datasets


def load_arc2(
    data_dir: Union[str, Path] = "data/arc-agi-2",
    splits: List[str] = ["training", "evaluation"]
) -> Dict[str, ARCDataset]:
    """
    Load ARC-AGI-2 dataset.
    
    Args:
        data_dir: Root directory of ARC-AGI-2 data
        splits: Which splits to load
    
    Returns:
        Dictionary mapping split name to ARCDataset
    """
    data_dir = Path(data_dir)
    datasets = {}
    
    for split in splits:
        split_dir = data_dir / split
        if split_dir.exists():
            datasets[split] = load_from_directory(split_dir, version="arc2", split=split)
        else:
            print(f"Warning: {split_dir} not found. Run scripts/download_data.py first.")
    
    return datasets


def load_all(data_dir: Union[str, Path] = "data") -> Dict[str, Dict[str, ARCDataset]]:
    """
    Load all available ARC datasets.
    
    Returns:
        Nested dictionary: version -> split -> ARCDataset
    """
    data_dir = Path(data_dir)
    
    return {
        "arc1": load_arc1(data_dir / "arc-agi-1"),
        "arc2": load_arc2(data_dir / "arc-agi-2"),
    }
