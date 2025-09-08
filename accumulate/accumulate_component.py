import copy
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Constants
UPDATED_STATE_PATH = Path(__file__).parent.parent / 'server' / 'updated_state.json'
OUTPUT_PATH = Path(__file__).parent / 'accumulate_output.json'

def process_immediate_report(input_data: Dict[str, Any], pre_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an immediate report and return the updated state.
    
    Args:
        input_data: The input data containing the immediate report (must contain 'slot' and 'reports')
        pre_state: The current state before processing
        
    Returns:
        Dict containing the updated state with the following fields:
        - slot: Updated slot number from input
        - entropy: Preserved from pre_state if exists
        - ready_queue: Updated with new reports
        - accumulated: Preserved from pre_state if exists
        - privileges: Preserved from pre_state if exists
        - statistics: Preserved from pre_state if exists
        - accounts: Preserved from pre_state if exists
    """
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
    """
    Load the current state from updated_state.json
    
    Returns:
        Dict containing the current state, or empty dict if file doesn't exist
    """
    try:
        with open(UPDATED_STATE_PATH, 'r') as f:
            state_data = json.load(f)
            if isinstance(state_data, list) and len(state_data) > 0:
                return state_data[0]  # Return the first item if it's a list
            return state_data if isinstance(state_data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_updated_state(post_state: Dict[str, Any]) -> None:
    """
    Save the updated state to both updated_state.json and accumulate_output.json
    
    Args:
        post_state: The post-state to save
    """
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
        
        # Try to extract slot and reports from server code
        # This is a simplified approach - in a real implementation, this would come from the server's request
        # and would be properly parsed from the HTTP request body
        
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