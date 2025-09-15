#!/usr/bin/env python3
"""
Test script to verify the jam-preimages integration.
"""
import os
import sys
import json
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sample preimage data
SAMPLE_PREIMAGES = [
    {
        "requester": 3,
        "blob": "0x92cdf578c47085a5992256f0dcf97d0b19f1f1c9de4d5fe30c3ace6191b6e5dbcee1b3419782ad92ec2dffed6d3f"
    }
]

def test_jam_preimages():
    """Test the jam-preimages component with sample data."""
    try:
        # Create a temporary input file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp_file:
            input_data = {
                "preimages": SAMPLE_PREIMAGES,
                "pre_state": {}
            }
            json.dump(input_data, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # Get the path to the main script
            jam_preimages_dir = os.path.dirname(os.path.abspath(__file__))
            main_script = os.path.join(jam_preimages_dir, "main.py")
            
            # Run the script with the input file
            cmd = ["python3", main_script, "--input", temp_file_path]
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=jam_preimages_dir,
                capture_output=True,
                text=True
            )
            
            # Print the output
            print("\n=== STDOUT ===")
            print(result.stdout)
            
            if result.stderr:
                print("\n=== STDERR ===")
                print(result.stderr)
            
            # Check the result
            if result.returncode != 0:
                print(f"\n❌ Test failed with return code {result.returncode}")
                return False
            
            # Check if the output contains the expected success message
            if "Successfully updated" in result.stdout:
                print("\n✅ Test passed!")
                return True
            else:
                print("\n❌ Test failed - expected success message not found")
                return False
                
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary file: {e}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_jam_preimages()
