"""
JAM Authorization Component Integration

This module provides authorization processing that:
1. Takes input from payload
2. Reads pre_state from updated_state.json when needed
3. Processes authorization using STF logic
4. Stores post_state back to updated_state.json
"""

import json
import os
import httpx
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from copy import deepcopy
from hashlib import sha256
import nacl.signing
import nacl.encoding
import secrets

# Simple SCALE encoding for PVM communication
def encode_u32(value: int) -> bytes:
    """Encode u32 as little-endian bytes."""
    return value.to_bytes(4, 'little')

def encode_u64(value: int) -> bytes:
    """Encode u64 as little-endian bytes."""
    return value.to_bytes(8, 'little')

def encode_bytes(data: bytes) -> bytes:
    """Encode bytes with length prefix."""
    return encode_u32(len(data)) + data

def encode_auth_credentials(public_key: bytes, signature: bytes, nonce: int) -> bytes:
    """Encode AuthCredentials for PVM."""
    if len(public_key) != 32:
        raise ValueError("Public key must be 32 bytes")
    if len(signature) != 64:
        raise ValueError("Signature must be 64 bytes")
    
    return public_key + signature + encode_u64(nonce)

def encode_work_package(payload: bytes, service_id: int = 1) -> bytes:
    """Encode WorkPackage for PVM with proper item structure."""
    # Create work item with proper SCALE encoding
    work_item = (
        encode_u32(service_id) +  # service_id
        b'\x00' * 32 +           # code_hash (32 bytes)
        encode_bytes(payload) +   # payload with length prefix
        encode_u64(1000000) +     # refine_gas
        encode_u64(1000000) +     # accumulate_gas  
        encode_u32(0) +           # export_count
        encode_u32(0) +           # imports count (empty vec)
        encode_u32(0)             # extrinsics count (empty vec)
    )
    
    # Work package structure with proper context
    context = (
        b'\x00' * 32 +           # anchor_hash
        b'\x00' * 32 +           # state_root
        b'\x00' * 32 +           # acc_output_log_peak
        b'\x00' * 32 +           # lookup_anchor_hash
        encode_u32(0) +           # lookup_timeslot
        encode_u32(0)             # prerequisites count (empty set)
    )
    
    return (
        encode_bytes(b'') +       # auth_token (empty)
        encode_u32(0) +           # auth_service_id
        b'\x00' * 32 +           # auth_code_hash
        encode_bytes(b'') +       # auth_config (empty)
        context +                 # refinement context
        encode_u32(1) +           # items count (1 item)
        work_item                 # the work item
    )

class AuthorizationProcessor:
    """Handles authorization processing with state management and PVM integration."""
    
    def __init__(self, server_dir: str = None, pvm_url: str = "http://127.0.0.1:8080"):
        self.server_dir = server_dir or os.path.dirname(__file__)
        self.state_file = os.path.join(self.server_dir, "updated_state.json")
        self.pvm_url = pvm_url
        
    def load_state(self) -> Dict[str, Any]:
        """Load current state from updated_state.json."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    
                # Handle list format
                if isinstance(state, list) and len(state) > 0:
                    state = state[0]
                    
                return state
            return self._get_default_state()
        except Exception as e:
            print(f"Error loading state: {e}")
            return self._get_default_state()
    
    def save_state(self, auth_state: Dict[str, Any]) -> bool:
        """Save authorization state back to updated_state.json while preserving existing data."""
        try:
            # Load the current complete state
            current_state = {}
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    current_state = json.load(f)
                    
                # Handle list format
                if isinstance(current_state, list) and len(current_state) > 0:
                    current_state = current_state[0]
            
            # Update only authorization-related fields while preserving everything else
            current_state.update({
                "auth_pools": auth_state.get("auth_pools", current_state.get("auth_pools", [])),
                "auth_queues": auth_state.get("auth_queues", current_state.get("auth_queues", [])),
                "authorizations": auth_state.get("authorizations", current_state.get("authorizations", {})),
                "slot": auth_state.get("slot", current_state.get("slot", 0))
            })
            
            # Update metadata to track authorization component updates
            if "metadata" not in current_state:
                current_state["metadata"] = {}
            
            current_state["metadata"].update({
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "updated_by": "authorization_processor",
                "auth_fields_updated": len([k for k in ["auth_pools", "auth_queues", "authorizations", "slot"] 
                                          if k in auth_state])
            })
            
            # Save the complete state back to file
            with open(self.state_file, 'w') as f:
                json.dump(current_state, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def _get_default_state(self) -> Dict[str, Any]:
        """Get default authorization state structure."""
        return {
            "auth_pools": [],
            "auth_queues": [],
            "authorizations": {},
            "slot": 0,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    async def process_authorization(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process authorization request using STF logic with PVM integration.
        
        Args:
            input_data: Authorization input containing:
                - slot: Current time slot
                - auths: List of authorization requests with core and auth_hash
                - public_key: Public key of requester
                - signature: Signature for verification
                - payload: Authorization payload
                
        Returns:
            Dict containing success status and updated state
        """
        try:
            # Load current pre_state
            pre_state = self.load_state()
            
            # Extract authorization pools and queues from pre_state
            auth_pools = deepcopy(pre_state.get("auth_pools", []))
            auth_queues = deepcopy(pre_state.get("auth_queues", []))
            authorizations = deepcopy(pre_state.get("authorizations", {}))
            
            # Process the authorization input
            slot = input_data.get("slot", pre_state.get("slot", 0))
            auths = input_data.get("auths", [])
            public_key = input_data.get("public_key")
            signature = input_data.get("signature")
            payload = input_data.get("payload", {})
            
            # PVM Authorization Verification
            pvm_authorized = False
            pvm_response = None
            
            if public_key and signature:
                try:
                    pvm_authorized, pvm_response = await self._verify_with_pvm(
                        public_key, signature, payload, slot
                    )
                except Exception as pvm_error:
                    print(f"PVM verification failed: {pvm_error}")
                    # Continue with local processing even if PVM fails
            
            # Apply STF logic for authorization processing
            post_state = self._apply_authorization_stf(
                auth_pools, auth_queues, authorizations, slot, auths, 
                public_key, payload, pvm_authorized, pvm_response
            )
            
            # Save post_state back to updated_state.json
            success = self.save_state(post_state)
            
            return {
                "success": success,
                "message": "Authorization processed successfully" if success else "Failed to save state",
                "pre_state": pre_state,
                "post_state": post_state,
                "slot": slot,
                "pvm_authorized": pvm_authorized,
                "pvm_response": pvm_response
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Authorization processing failed: {str(e)}",
                "error": str(e)
            }
    
    async def _verify_with_pvm(
        self, 
        public_key: str, 
        signature: str, 
        payload: dict, 
        slot: int
    ) -> tuple[bool, dict]:
        """
        Verify authorization with PVM using proper SCALE encoding.
        """
        try:
            # Get nonce for this public key
            state = self.load_state()
            current_auth = state.get("authorizations", {}).get(public_key, {})
            nonce = current_auth.get("nonce", 0)
            
            # Prepare payload data - hash the payload like PVM expects
            payload_json = json.dumps(payload, sort_keys=True)
            payload_data = payload_json.encode('utf-8')
            
            # Convert hex strings to bytes
            public_key_bytes = bytes.fromhex(public_key)
            signature_bytes = bytes.fromhex(signature)
            
            # Encode auth credentials using simple SCALE encoding
            encoded_auth = encode_auth_credentials(public_key_bytes, signature_bytes, nonce)
            
            # Encode work package
            encoded_package = encode_work_package(payload_data, payload.get('service_id', 1))
            
            # Make PVM request
            pvm_request = {
                "param_hex": encoded_auth.hex(),
                "package_hex": encoded_package.hex(),
                "core_index_hex": "00000000"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.pvm_url}/authorizer/is_authorized",
                    json=pvm_request,
                    timeout=10.0
                )
                response.raise_for_status()
                pvm_result = response.json()
            
            # Check if authorization was successful
            # PVM returns the auth credentials hex if successful
            success = pvm_result.get("output_hex") == encoded_auth.hex()
            
            return success, {
                "pvm_result": pvm_result,
                "nonce": nonce,
                "auth_credentials_hex": encoded_auth.hex(),
                "work_package_hex": encoded_package.hex()
            }
            
        except Exception as e:
            print(f"PVM verification error: {e}")
            return False, {"error": str(e)}
    
    def _apply_authorization_stf(
        self, 
        auth_pools: list, 
        auth_queues: list, 
        authorizations: dict,
        slot: int,
        auths: list,
        public_key: str,
        payload: dict,
        pvm_authorized: bool = False,
        pvm_response: dict = None
    ) -> Dict[str, Any]:
        """
        Apply State Transition Function (STF) logic for authorization.
        
        This implements the core authorization logic similar to the STF in 
        authorizations/full_cycle_test.py but adapted for direct payload processing.
        """
        # Ensure pools and queues have sufficient capacity
        max_cores = max(len(auth_pools), len(auths)) if auths else len(auth_pools)
        while len(auth_pools) < max_cores:
            auth_pools.append([])
        while len(auth_queues) < max_cores:
            auth_queues.append([])
        
        # Process authorization requests
        updated_cores = set()
        
        if auths:
            for auth in auths:
                core = auth.get("core", 0)
                auth_hash = auth.get("auth_hash", "")
                
                if core < len(auth_pools):
                    # Remove existing auth_hash if present
                    if auth_hash in auth_pools[core]:
                        auth_pools[core].remove(auth_hash)
                    
                    # Add new authorization to pool (max 8 entries)
                    if len(auth_pools[core]) >= 8:
                        auth_pools[core].pop(0)
                    auth_pools[core].append(auth_hash)
                    
                    # Add to queue if not already present
                    if auth_hash not in auth_queues[core]:
                        auth_queues[core].append(auth_hash)
                    
                    updated_cores.add(core)
        
        # Process queues for cores not explicitly updated
        for core in range(len(auth_pools)):
            if core in updated_cores:
                continue
                
            if auth_queues[core]:
                # Move from queue to pool
                hash_to_move = auth_queues[core].pop(0)
                
                if hash_to_move in auth_pools[core]:
                    auth_pools[core].remove(hash_to_move)
                if len(auth_pools[core]) >= 8:
                    auth_pools[core].pop(0)
                auth_pools[core].append(hash_to_move)
        
        # Update authorization records if public_key provided
        if public_key:
            # Get or initialize nonce
            current_auth = authorizations.get(public_key, {})
            nonce = current_auth.get("nonce", 0) + 1
            
            # Update authorization record with PVM status
            authorizations[public_key] = {
                "public_key": public_key,
                "nonce": nonce,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
                "slot": slot,
                "pvm_authorized": pvm_authorized,
                "pvm_response": pvm_response
            }
        
        # Return only the authorization-specific updates
        return {
            "auth_pools": auth_pools,
            "auth_queues": auth_queues,
            "authorizations": authorizations,
            "slot": slot
        }
    
    def get_authorization_status(self, public_key: str) -> Dict[str, Any]:
        """Get current authorization status for a public key."""
        state = self.load_state()
        auth_data = state.get("authorizations", {}).get(public_key, {})
        
        return {
            "authorized": bool(auth_data),
            "nonce": auth_data.get("nonce", 0),
            "last_updated": auth_data.get("last_updated"),
            "payload": auth_data.get("payload", {}),
            "slot": auth_data.get("slot", 0)
        }
    
    def create_auth_hash(self, public_key: str, payload: dict, slot: int) -> str:
        """Create authorization hash from public key, payload, and slot."""
        auth_data = {
            "public_key": public_key,
            "payload": payload,
            "slot": slot
        }
        auth_string = json.dumps(auth_data, sort_keys=True)
        return sha256(auth_string.encode()).hexdigest()
    
    def create_ed25519_keypair(self, seed: str = None) -> tuple[str, str]:
        """Create Ed25519 keypair for testing."""
        if seed:
            # Use provided seed
            seed_bytes = bytes.fromhex(seed) if len(seed) == 64 else seed.encode()[:32]
            if len(seed_bytes) < 32:
                seed_bytes = seed_bytes + b'\x00' * (32 - len(seed_bytes))
            private_key = nacl.signing.SigningKey(seed_bytes)
        else:
            # Generate random keypair
            private_key = nacl.signing.SigningKey.generate()
        
        public_key = private_key.verify_key
        
        return public_key.encode().hex(), private_key.encode().hex()
    
    def sign_payload(self, payload: dict, private_key_hex: str) -> str:
        """Sign payload with Ed25519 private key."""
        private_key = nacl.signing.SigningKey(bytes.fromhex(private_key_hex))
        payload_json = json.dumps(payload, sort_keys=True)
        payload_hash = sha256(payload_json.encode()).digest()
        signature = private_key.sign(payload_hash)
        return signature.signature.hex()

# Main processor instance
authorization_processor = AuthorizationProcessor()
