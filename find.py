#!/usr/bin/env python3
import os

# Traverse the directory to find "Atelectasis" and "Normal"
def find_subdirectories(base_dir):
    atelectasis_path = None
    normal_path = None

    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if "Atelectasis" in dir_name and atelectasis_path is None:
                atelectasis_path = os.path.join(root, dir_name)
            if "Normal" in dir_name and normal_path is None:
                normal_path = os.path.join(root, dir_name)
        
        # If both directories are found, no need to continue the search
        if atelectasis_path and normal_path:
            break

    return atelectasis_path, normal_path

# Example usage:
base_dir = "./data/data1"  # Replace with your actual base directory
atelectasis_path, normal_path = find_subdirectories(base_dir)

if atelectasis_path and normal_path:
    print("Atelectasis directory:", atelectasis_path)
    print("Normal directory:", normal_path)
else:
    print("Error: Could not find both 'Atelectasis' and 'Normal' directories.")

