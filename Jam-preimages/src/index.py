import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any
from .stf.run_test import run_preimage_test
from .utils.json_loader import load_test_vector


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(directory, exist_ok=True)


def save_results(test_name: str, result: Dict[str, Any], output_dir: str) -> None:
    """Save test results to a JSON file."""
    output_file = os.path.join(output_dir, test_name)
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)


def main():
    """Main function to run JAM Preimage STF Tests and generate results."""
    # Setup directories
    current_dir = Path(__file__).parent
    results_dir = current_dir.parent / 'results'
    test_vectors_dir = current_dir.parent / 'test-vectors'
    
    # Clean and create results directory
    if results_dir.exists():
        shutil.rmtree(results_dir)
    ensure_dir(results_dir)
    
    # Test directories to process
    test_dirs = ["tiny", "full"]
    passed = 0
    failed = 0
    
    print(" ✅ ✅ ✅ Running JAM Preimage:  ✅ ✅ ✅ \n")
    
    for test_dir in test_dirs:
        test_dir_path = test_vectors_dir / test_dir
        if not test_dir_path.exists():
            print(f"Warning: Test directory {test_dir_path} does not exist")
            continue
            
        # Create subdirectory in results
        test_results_dir = results_dir / test_dir
        ensure_dir(test_results_dir)
        
        # Get all JSON test files
        test_files = [f for f in os.listdir(test_dir_path) if f.endswith('.json')]
        
        print(f" ✅  Running tests in {test_dir}:")
        
        for test_file in test_files:
            try:
                # Load test vector
                test = load_test_vector(test_dir, test_file)
                test.name = test_file
                
                # Run test
                result = run_preimage_test(test)
                
                # Save results
                save_results(test_file, result, str(test_results_dir))
                
                # Update counters
                if result.get('verified', False):
                    print(f" {test_file}: Passed")
                    passed += 1
                else:
                    print(f" {test_file}: Failed")
                    failed += 1
                    
            except Exception as e:
                print(f" {test_file}: Threw Error")
                print(f"  {str(e)}")
                failed += 1
    
    # Print summary
    total = passed + failed
    print(f"\nTest Results:")
    print(f"  ✅ Passed: {passed}")
    print(f" Failed: {failed}")
    print(f" Total:  {total}")
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
