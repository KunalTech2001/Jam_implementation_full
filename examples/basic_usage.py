#!/usr/bin/env python3
"""
Basic usage example for JAM Protocol.

This example demonstrates how to use the SafroleManager
to process blocks and manage state transitions.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jam import SafroleManager


def main():
    """Demonstrate basic JAM protocol usage."""
    
    initial_state = {
        "tau": 0,
        "eta": [
            "0x0000000000000000000000000000000000000000000000000000000000000000",
            "0x0000000000000000000000000000000000000000000000000000000000000000",
            "0x0000000000000000000000000000000000000000000000000000000000000000",
            "0x0000000000000000000000000000000000000000000000000000000000000000"
        ],
        "gamma_k": [
            {
                "bandersnatch": "0x1ecc3686b60ee3b84b6c7d321d70d5c06e9dac63a4d0a79d731b17c0d04d030d",
                "ed25519": "0x1ecc3686b60ee3b84b6c7d321d70d5c06e9dac63a4d0a79d731b17c0d04d030d",
                "bls": "0x" + "0" * 288,
                "metadata": "0x" + "0" * 256
            }
        ],
        "gamma_z": "0x" + "0" * 64,
        "lambda": [],
        "kappa": [],
        "iota": [],
        "gamma_a": [],
        "post_offenders": []
    }
    
    manager = SafroleManager(initial_state)
    
    print("Initial state created successfully!")
    print(f"Current tau: {manager.state['tau']}")
    print(f"Current eta: {manager.state['eta']}")
    
    block_input = {
        "slot": 1,
        "entropy": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "extrinsic": []
    }
    
    try:
        result = manager.process_block(block_input)
        
        print("\nBlock processed successfully!")
        print(f"New tau: {result['post_state']['tau']}")
        print(f"Header slot: {result['header']['slot']}")
        print(f"VRF output: {result['header']['vrf_output']}")
        
    except Exception as e:
        print(f"Error processing block: {e}")


if __name__ == "__main__":
    main() 