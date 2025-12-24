"""
Download ARC-AGI dataset from GitHub repository.
"""

import os
import json
import urllib.request
from pathlib import Path

REPO_BASE = "https://raw.githubusercontent.com/fchollet/ARC-AGI/master/data"

def download_file(url, save_path):
    """Download a file from URL to save_path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        with urllib.request.urlopen(url) as response:
            content = response.read()
            with open(save_path, 'wb') as f:
                f.write(content)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def get_task_list():
    """Get list of all task files from the repository."""
    # Known task IDs - you can expand this list or fetch from the repo
    # For now, we'll use a direct API call to get the file list
    
    training_url = f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/training"
    evaluation_url = f"https://api.github.com/repos/fchollet/ARC-AGI/contents/data/evaluation"
    
    training_tasks = []
    evaluation_tasks = []
    
    try:
        # Get training tasks
        with urllib.request.urlopen(training_url) as response:
            files = json.loads(response.read())
            training_tasks = [f['name'] for f in files if f['name'].endswith('.json')]
        
        # Get evaluation tasks  
        with urllib.request.urlopen(evaluation_url) as response:
            files = json.loads(response.read())
            evaluation_tasks = [f['name'] for f in files if f['name'].endswith('.json')]
            
    except Exception as e:
        print(f"Error fetching task list: {e}")
        
    return training_tasks, evaluation_tasks

def download_dataset():
    """Download the complete ARC-AGI dataset."""
    
    print("Fetching task list from GitHub...")
    training_tasks, evaluation_tasks = get_task_list()
    
    print(f"Found {len(training_tasks)} training tasks")
    print(f"Found {len(evaluation_tasks)} evaluation tasks")
    
    # Download training tasks
    print("\nDownloading training tasks...")
    for task_file in training_tasks:
        url = f"{REPO_BASE}/training/{task_file}"
        save_path = f"data/training/{task_file}"
        
        if download_file(url, save_path):
            print(f"  ✓ {task_file}")
        else:
            print(f"  ✗ {task_file}")
    
    # Download evaluation tasks
    print("\nDownloading evaluation tasks...")
    for task_file in evaluation_tasks:
        url = f"{REPO_BASE}/evaluation/{task_file}"
        save_path = f"data/evaluation/{task_file}"
        
        if download_file(url, save_path):
            print(f"  ✓ {task_file}")
        else:
            print(f"  ✗ {task_file}")
    
    print(f"\n✅ Dataset download complete!")
    print(f"   Training tasks: data/training/ ({len(training_tasks)} files)")
    print(f"   Evaluation tasks: data/evaluation/ ({len(evaluation_tasks)} files)")

if __name__ == "__main__":
    download_dataset()
