
#!/usr/bin/env python3
"""Storage cleanup utility for Replit environment"""

import os
import shutil
import subprocess
from pathlib import Path

def cleanup_storage():
    """Clean up storage to free space"""
    print("Starting storage cleanup...")
    
    # 1. Clean Python cache
    print("Cleaning Python cache...")
    for root, dirs, files in os.walk("."):
        for d in dirs[:]:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                dirs.remove(d)
    
    # 2. Clean pip cache
    print("Cleaning pip cache...")
    try:
        subprocess.run(["pip", "cache", "purge"], check=False)
    except Exception:
        pass
    
    # 3. Clean temporary files
    print("Cleaning temporary files...")
    temp_patterns = ["*.tmp", "*.temp", "*.cache"]
    for pattern in temp_patterns:
        try:
            subprocess.run(["find", ".", "-name", pattern, "-delete"], check=False)
        except Exception:
            pass
    
    # 4. Clean old logs
    print("Cleaning old logs...")
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            if log_file.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                log_file.unlink(missing_ok=True)
    
    # 5. Clean large workspace files
    print("Checking workspace for large files...")
    workspace_dir = Path("workspace")
    if workspace_dir.exists():
        for file in workspace_dir.rglob("*"):
            if file.is_file() and file.stat().st_size > 50 * 1024 * 1024:  # > 50MB
                print(f"Large file found: {file} ({file.stat().st_size / 1024 / 1024:.1f}MB)")
    
    print("Storage cleanup completed!")

if __name__ == "__main__":
    cleanup_storage()
