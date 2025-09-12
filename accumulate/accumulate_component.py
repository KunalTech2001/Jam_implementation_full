import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PVMConfig:
    def __init__(self):
        self.base_url = os.getenv("PVM_BASE_URL", "http://127.0.0.1:8080")
        self.timeout = int(os.getenv("PVM_TIMEOUT_SECONDS", "15"))
        self.max_retries = int(os.getenv("PVM_MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("PVM_RETRY_DELAY_SECONDS", "0.5"))

# Custom exceptions
class PVMError(Exception):
    """Base exception for PVM-related errors"""
    pass

class PVMConnectionError(PVMError):
    """Raised when unable to connect to PVM"""
    pass

class PVMResponseError(PVMError):
    """Raised when PVM returns an error response"""
    pass

# Global config instance
pvm_config = PVMConfig()

# Constants
UPDATED_STATE_PATH = Path(__file__).parent.parent / 'server' / 'updated_state.json'
OUTPUT_PATH = Path(__file__).parent / 'accumulate_output.json'

def process_immediate_report(input_data: Dict[str, Any], pre_state: Dict[str, Any]) -> Dict[str, Any]:

    # Create a deep copy of the pre_state to avoid modifying it directly
    post_state = copy.deepcopy(pre_state)
    
    # Update the slot number from input
    post_state["slot"] = input_data["slot"]
    
    # Preserve the entropy from pre_state if it exists
    if "entropy" in pre_state:
        post_state["entropy"] = pre_state["entropy"]
    
    # Initialize ready_queue if it doesn't exist (12 cores)
    if "ready_queue" not in post_state:
        post_state["ready_queue"] = [[] for _ in range(12)]
    
    # Ensure ready_queue has exactly 12 cores
    while len(post_state["ready_queue"]) < 12:
        post_state["ready_queue"].append([])
    post_state["ready_queue"] = post_state["ready_queue"][:12]
    
    # Process each report in the input
    for report in input_data.get("reports", []):
        # Get the core index, default to 0 if not specified
        core_index = report.get("core_index", 0)
        
        # Ensure the core index is within bounds (0-11)
        if 0 <= core_index < 12:
            # Ensure the core's queue is a list
            if not isinstance(post_state["ready_queue"][core_index], list):
                post_state["ready_queue"][core_index] = []
            
            # Add the report to the appropriate queue with its dependencies
            post_state["ready_queue"][core_index].append({
                "report": report,
                "dependencies": report.get("prerequisites", [])
            })
    
    # Preserve other important fields if they exist in pre_state
    for field in ["accumulated", "privileges", "statistics", "accounts"]:
        if field in pre_state:
            post_state[field] = pre_state[field]
    
    # Ensure accumulated has 12 slots
    if "accumulated" in post_state:
        while len(post_state["accumulated"]) < 12:
            post_state["accumulated"].append([])
        post_state["accumulated"] = post_state["accumulated"][:12]
    
    return post_state

def load_updated_state() -> Dict[str, Any]:

    try:
        with open(UPDATED_STATE_PATH, 'r') as f:
            state_data = json.load(f)
            if isinstance(state_data, list) and len(state_data) > 0:
                return state_data[0]  # Return the first item if it's a list
            return state_data if isinstance(state_data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_updated_state(post_state: Dict[str, Any]) -> None:
  
    # Ensure the output directory exists
    UPDATED_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to both locations
    for path in [UPDATED_STATE_PATH, OUTPUT_PATH]:
        with open(path, 'w') as f:
            json.dump([post_state] if path == UPDATED_STATE_PATH else post_state, f, indent=2)

def process_immediate_report_from_server() -> Optional[Dict[str, Any]]:
    try:
        # Load input from server payload (this would come from the server's POST request)
        # For now, we'll load it from the server.py file
        server_path = Path(__file__).parent.parent / 'server' / 'server.py'
        with open(server_path, 'r') as f:
            server_code = f.read()
        
        # Extract the payload from server.py (this is a simplified approach)
        # In a real implementation, this would come from the server's request
        input_data = {
            "slot": 43,  # Default value if not found
            "reports": []
        }
     
        # Load pre_state
        pre_state = load_updated_state()
        
        # Process the immediate report
        post_state = process_immediate_report(input_data, pre_state)
        
        # Save the updated state
        save_updated_state(post_state)
        
        return post_state
        
    except Exception as e:
        print(f"Error processing immediate report: {e}", file=sys.stderr)
        return None

    current_ready = post_state['ready_queue'][cur]

    # Process input reports
    for rpt in shallow_flatten(input.get('reports', [])):
        if not isinstance(rpt, dict):
            continue
        deps = set(rpt.get('context', {}).get('prerequisites', []))
        for item in rpt.get('segment_root_lookup', []):
            if isinstance(item, dict):
                deps.add(item.get('hash', ''))
        deps -= hashes

        pkg = rpt.get('package_spec', {})
        pkg_h = pkg.get('hash', '') if isinstance(pkg, dict) else ''

        res = rpt.get('results', [])
        if isinstance(res, list) and res and isinstance(res[0], dict):
            r0 = res[0]
            ok = isinstance(r0.get('result', {}), dict) and r0['result'].get('ok') is not None
            gas = r0.get('accumulate_gas', 0)
            svc = r0.get('service_id')
        else:
            ok = False
            gas = 0
            svc = None

        auth_gas = rpt.get('auth_gas_used', 0)

        aff = False
        if svc is not None and gas > 0:
            for a in post_state.get('accounts', []):
                if isinstance(a, dict) and a.get('id') == svc:
                    balance = a.get('data', {}).get('service', {}).get('balance', 0)
                    if balance >= gas:
                        aff = True
                    break

        if svc and ok and aff and not deps and pkg_h:
            acc.append(pkg_h)
            hashes.add(pkg_h)
            if 'statistics' not in post_state:
                post_state['statistics'] = []
            stats = next((x for x in post_state['statistics'] if isinstance(x, dict) and x.get('service_id') == svc), None)
            if stats is None:
                stats = {'service_id': svc, 'accumulate_count': 0, 'accumulate_gas_used': 0, 'on_transfers_count': 0, 'on_transfers_gas_used': 0, 'record': {'provided_count': 0, 'provided_size': 0, 'refinement_count': 0, 'refinement_gas_used': 0, 'imports': 0, 'exports': 0, 'extrinsic_size': 0, 'extrinsic_count': 0, 'accumulate_count': 0, 'accumulate_gas_used': 0, 'on_transfers_count': 0, 'on_transfers_gas_used': 0}}
                post_state['statistics'].append(stats)
            stats['accumulate_count'] += 1
            stats['accumulate_gas_used'] += gas + auth_gas
            stats['record']['accumulate_count'] += 1
            stats['record']['accumulate_gas_used'] += gas + auth_gas
            for a in post_state.get('accounts', []):
                if isinstance(a, dict) and a.get('id') == svc:
                    a['data']['service']['balance'] -= gas
                    break

        current_ready.append({'report': rpt, 'dependencies': list(deps), 'stale': pkg_h in deps})

    return {'ok': merkle_root(acc)}, post_state

# ---------------- New integration helpers for jam_pvm ----------------

JAM_PVM_BASE = "http://127.0.0.1:8080"
ACCUMULATE_JSON_ENDPOINT = f"{JAM_PVM_BASE}/service/accumulate_json"

def bytes_sha256_hex(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def build_accumulate_item_json(
    auth_output_hex: str,
    payload_bytes: bytes,
    ok: bool = True,
    work_output_bytes: Optional[bytes] = None,
    package_hash_hex: str = "00" * 32,
    exports_root_hex: str = "00" * 32,
    authorizer_hash_hex: str = "00" * 32,
) -> Dict[str, Any]:
    """
    Build a JSON item compatible with jam_pvm `/service/accumulate_json`.
    - payload hash is computed as sha256(payload_bytes)
    - work_output_hex is required when ok=True
    """
    item: Dict[str, Any] = {
        "auth_output_hex": auth_output_hex,
        "payload_hash_hex": bytes_sha256_hex(payload_bytes),
        "result_ok": bool(ok),
        "work_output_hex": work_output_bytes.hex() if (ok and work_output_bytes is not None) else None,
        "package_hash_hex": package_hash_hex,
        "exports_root_hex": exports_root_hex,
        "authorizer_hash_hex": authorizer_hash_hex,
    }
    if ok and item["work_output_hex"] is None:
        raise ValueError("work_output_bytes must be provided when ok=True")
    return item

def post_accumulate_json_with_retry(
    slot: int, 
    service_id: int, 
    items: List[Dict[str, Any]], 
    config: Optional[PVMConfig] = None
) -> Dict[str, Any]:
    """
    Post accumulate data to jam_pvm JSON endpoint with retry logic.
    
    Args:
        slot: The slot number
        service_id: The service ID
        items: List of dicts built by build_accumulate_item_json
        config: Optional PVM configuration
        
    Returns:
        Dict containing the PVM response
        
    Raises:
        PVMConnectionError: If unable to connect to PVM after retries
        PVMResponseError: If PVM returns an error response
    """
    config = config or pvm_config
    endpoint = f"{config.base_url}/service/accumulate_json"
    payload = {
        "slot": int(slot),
        "service_id": int(service_id),
        "items": items,
    }
    
    last_error = None
    for attempt in range(config.max_retries + 1):
        try:
            resp = requests.post(
                endpoint,
                json=payload,
                timeout=config.timeout
            )
            resp.raise_for_status()
            return resp.json()
            
        except requests.RequestException as e:
            last_error = e
            if attempt < config.max_retries:
                delay = config.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Attempt {attempt + 1}/{config.max_retries} failed. "
                    f"Retrying in {delay:.2f}s. Error: {str(e)}"
                )
                time.sleep(delay)
                continue
            raise PVMConnectionError(
                f"Failed to connect to PVM after {config.max_retries} attempts: {str(e)}"
            ) from e
        except Exception as e:
            raise PVMResponseError(f"Error processing PVM response: {str(e)}") from e

def process_with_pvm(
    input_data: Dict[str, Any],
    pre_state: Dict[str, Any],
    config: Optional[PVMConfig] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process input data with PVM integration using dynamic payload.
    
    Args:
        input_data: Dynamic input data containing reports and other fields
        pre_state: Current state from the system
        config: Optional PVM configuration
        
    Returns:
        Tuple of (updated_state, pvm_responses)
    """
    # 1. Process the immediate report to get updated state
    post_state = process_immediate_report(input_data, pre_state)
    
    # 2. Extract PVM-related data from input payload
    slot = input_data.get("slot")
    if slot is None:
        logger.warning("No slot provided in input_data, using current slot from state")
        slot = pre_state.get("slot")
    
    # 3. Prepare PVM items from input reports
    items = []
    for report in input_data.get("reports", []):
        try:
            # Get service_id from report results if available
            service_id = None
            results = report.get("results", [])
            if results and isinstance(results[0], dict):
                service_id = results[0].get("service_id")
            
            # Get package hash if available
            package_hash = "00" * 32
            if "package_spec" in report and isinstance(report["package_spec"], dict):
                package_hash = report["package_spec"].get("hash", package_hash)
            
            # Get work output (result) if available
            work_output = None
            if results and isinstance(results[0], dict) and "result" in results[0]:
                result = results[0]["result"]
                if isinstance(result, dict) and "ok" in result:
                    work_output = result["ok"]
                    if work_output and isinstance(work_output, str):
                        work_output = work_output.removeprefix("0x").encode()
            
            # Build the PVM item
            payload_bytes = json.dumps(report, sort_keys=True).encode("utf-8")
            item = build_accumulate_item_json(
                auth_output_hex="00",  # Should be replaced with actual auth
                payload_bytes=payload_bytes,
                ok=work_output is not None,
                work_output_bytes=work_output,
                package_hash_hex=package_hash,
                exports_root_hex="00" * 32,  # Should be replaced with actual exports root
                authorizer_hash_hex="00" * 32  # Should be replaced with actual authorizer
            )
            items.append((service_id, item))
            
        except Exception as e:
            logger.error(f"Failed to prepare PVM item from report: {e}")
            continue
    
    # 4. Group items by service_id and send to PVM
    pvm_responses = {}
    service_items = {}
    
    # Group items by service_id
    for service_id, item in items:
        if service_id not in service_items:
            service_items[service_id] = []
        service_items[service_id].append(item)
    
    # Send each service's items to PVM
    for service_id, service_items_list in service_items.items():
        if service_id is None:
            logger.warning("Skipping items with no service_id")
            continue
            
        try:
            pvm_response = post_accumulate_json_with_retry(
                slot=slot,
                service_id=service_id,
                items=service_items_list,
                config=config
            )
            pvm_responses[service_id] = pvm_response
            logger.info(f"Successfully sent {len(service_items_list)} items to PVM for service {service_id}")
        except Exception as e:
            logger.error(f"Failed to send items to PVM for service {service_id}: {e}")
            pvm_responses[service_id] = {"error": str(e), "service_id": service_id}
    
    return post_state, pvm_responses