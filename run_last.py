#!/usr/bin/env python3
"""
run_last.py - Automatically run a demo of the best genome from the latest generation run.

This script uses the 'ga_checkpoints_v2/last_run.json' file which is continuously saved
during the training phase (in run.py), to easily verify if the robot from the last run 
can successfully exit the maze.
"""
import os
import sys
import subprocess

def main():
    ckpt = "ga_checkpoints_v2/last_run.json"
    
    if not os.path.exists(ckpt):
        print(f"Error: Could not find '{ckpt}'.")
        print("Please run the training ('run.py') for at least one generation first.")
        sys.exit(1)
        
    print(f"[*] Visualizing the best genome from the last run ({ckpt})")
    print("[*] Launching run.py in demo mode...")
    
    # Run the equivalent of: python3 run.py --run ga_checkpoints_v2/last_run.json
    cmd = ["python3", "run.py", "--run", ckpt]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n[!] run.py exited with error code {e.returncode}")

if __name__ == "__main__":
    main()
