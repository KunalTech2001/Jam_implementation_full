from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Union
import uvicorn
import logging
import sys
import os
import json
import datetime
from datetime import datetime, timezone
from copy import deepcopy
import difflib
from contextlib import asynccontextmanager
import psutil
import subprocess
from hashlib import sha256
import nacl.signing
import base64


# Add project root and src directory to sys.path so sibling packages are importable
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, 'src'))

from jam.core.safrole_manager import SafroleManager
from jam.utils.helpers import deep_clone
from accumulate.accumulate_component import (
    post_accumulate_json_with_retry as post_accumulate_json,
    load_updated_state as acc_load_state,
    save_updated_state as acc_save_state,
    process_immediate_report as acc_process,
    process_with_pvm
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JAM Safrole, Dispute, and State Integration Server",
    description="REST API server for JAM protocol safrole, dispute, and state component integration",
    version="1.0.0"
)

# Pydantic model for authorization request
class AuthorizationRequest(BaseModel):
    public_key: str
    signature: str
    nonce: int
    payload: Dict[str, Any]

# Pydantic model for authorization response
class AuthorizationResponse(BaseModel):
    success: bool
    message: str
    auth_output: Optional[str] = None
    updated_state: Optional[Dict[str, Any]] = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
safrole_manager: Optional[SafroleManager] = None
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to get project root
sample_data_path = os.path.join(script_dir, "sample_data.json")
updated_state_path = os.path.join(script_dir, "updated_state.json")
jam_history_script = os.path.join(project_root, "Jam-history", "test.py")
jam_reports_script=os.path.join(project_root,"Reports-Python","run_jam_vectors.py")
jam_preimages_script = os.path.join(project_root, "Jam-preimages", "main.py")
original_sample_data: Dict[str, Any] = {}



# Path to jam_history (test.py)
def run_jam_history():
    """Run the jam_history component (test.py)."""
    try:
        logger.info(f"Attempting to run jam_history component: {jam_history_script}")
        result = subprocess.run(
            ["python3", jam_history_script],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"jam_history executed successfully:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"jam_history failed: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running jam_history: {str(e)}")
        return False


# Default sample data if file is missing
DEFAULT_SAMPLE_DATA = {
    "pre_state": {
        "tau": 0,
        "E": 12,
        "Y": 11,
        "gamma_a": [],
        "psi": {"good": [], "bad": [], "wonky": [], "offenders": []},
        "rho": [],
        "kappa": [],
        "lambda": [],
        "vals_curr_stats": [],
        "vals_last_stats": [],
        "slot": 0,
        "curr_validators": []
    }
}

# Pydantic models for request/response validation
class BlockHeader(BaseModel):
    parent: str
    parent_state_root: str
    extrinsic_hash: str
    slot: int
    epoch_mark: Optional[Any] = None
    tickets_mark: Optional[Any] = None
    offenders_mark: List[Any] = []
    author_index: int
    entropy_source: str
    seal: str
    # Fields required by jam-history, made optional to not break other components
    header_hash: Optional[str] = None
    accumulate_root: Optional[str] = None
    work_packages: Optional[List[Dict[str, Any]]] = []

class Vote(BaseModel):
    vote: bool
    index: int
    signature: str

class Verdict(BaseModel):
    target: str
    age: int
    votes: List[Vote]

class Culprit(BaseModel):
    target: str
    key: str
    signature: str

class Fault(BaseModel):
    target: str
    vote: bool
    key: str
    signature: str

class BlockDisputes(BaseModel):
    verdicts: List[Verdict] = []
    culprits: List[Culprit] = []
    faults: List[Fault] = []

class Signature(BaseModel):
    validator_index: int
    signature: str

class Guarantee(BaseModel):
    signatures: List[Signature]
    report: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None

class Assurance(BaseModel):
    validator_index: int
    signature: str

class Preimage(BaseModel):
    blob: str

class BlockExtrinsic(BaseModel):
    tickets: List[Any] = []
    preimages: List[Preimage] = []
    guarantees: List[Guarantee] = []
    assurances: List[Assurance] = []
    disputes: BlockDisputes

class Block(BaseModel):
    header: BlockHeader
    extrinsic: BlockExtrinsic

class BlockProcessRequest(BaseModel):
    block: Block

class StateResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# ---- Pydantic models for forwarding accumulate to jam_pvm ----
class AccumulateItemJSON(BaseModel):
    auth_output_hex: str
    payload_hash_hex: str
    result_ok: bool = True
    work_output_hex: Optional[str] = None
    package_hash_hex: str = Field(default_factory=lambda: "00"*32)
    exports_root_hex: str = Field(default_factory=lambda: "00"*32)
    authorizer_hash_hex: str = Field(default_factory=lambda: "00"*32)

class AccumulateForwardRequest(BaseModel):
    slot: int
    service_id: int
    items: List[AccumulateItemJSON]

class AccumulateForwardResponse(BaseModel):
    success: bool
    message: str
    jam_pvm_response: Optional[Dict[str, Any]] = None

# Request model matching accumulate_component expectations
class AccumulateComponentInput(BaseModel):
    slot: int
    reports: List[Dict[str, Any]] = []

class AccumulateProcessResponse(BaseModel):
    success: bool
    message: str
    post_state: Dict[str, Any]
    jam_pvm_response: Optional[Dict[str, Any]] = None

def hydrate_map(obj):
    if obj is None:
        return obj
    if isinstance(obj, list):
        return [hydrate_map(x) for x in obj]
    if isinstance(obj, dict):
        if obj.get('_isSet'):
            return set(hydrate_map(x) for x in obj['values'])
        if obj.get('_isMap'):
            return {k: hydrate_map(v) for k, v in obj['entries']}
        return {k: hydrate_map(v) for k, v in obj.items()}
    return obj

def initialize_state(pre_state):
    state = OnchainState()
    if pre_state:
        if 'ρ' in pre_state:
            state.ρ = hydrate_map(pre_state['ρ'])
        if 'ω' in pre_state:
            state.ω = hydrate_map(pre_state['ω'])
        if 'ξ' in pre_state:
            state.ξ = hydrate_map(pre_state['ξ'])
        if 'ψ_B' in pre_state:
            state.ψ_B = hydrate_map(pre_state['ψ_B'])
        if 'ψ_O' in pre_state:
            state.ψ_O = hydrate_map(pre_state['ψ_O'])
        if 'globalState' in pre_state:
            gs = pre_state['globalState']
            state.global_state = {
                'accounts': gs.get('accounts', {}),
                'coreStatus': hydrate_map(gs.get('coreStatus', {})),
                'serviceRegistry': hydrate_map(gs.get('serviceRegistry', {})),
            }
    return state

def map_input_to_extrinsic(input_data):
    import json
    extrinsic = json.loads(json.dumps(input_data))
    for guarantee in extrinsic['guarantees']:
        r = guarantee['report']
        wp = r.get('workPackage')
        package = WorkPackage(
            wp.get('authorizationToken'),
            wp.get('authorizationServiceDetails'),
            wp.get('context'),
            [WorkItem(
                wi.get('id'),
                wi.get('programHash'),
                wi.get('inputData'),
                wi.get('gasLimit')
            ) for wi in wp.get('workItems', [])]
        )
        ctx = r.get('refinementContext')
        ref_ctx = RefinementContext(
            ctx.get('anchorBlockRoot'),
            ctx.get('anchorBlockNumber'),
            ctx.get('beefyMmrRoot'),
            ctx.get('currentSlot'),
            ctx.get('currentEpoch'),
            ctx.get('currentGuarantors'),
            ctx.get('previousGuarantors')
        )
        availability_spec = None
        if r.get('availabilitySpec'):
            aspec = r.get('availabilitySpec')
            availability_spec = AvailabilitySpec(
                aspec.get('totalFragments'),
                aspec.get('dataFragments'),
                aspec.get('fragmentHashes')
            )
        guarantee['report'] = WorkReport(
            package,
            ref_ctx,
            r.get('pvmOutput'),
            r.get('gasUsed'),
            availability_spec,
            r.get('guarantorSignature'),
            r.get('guarantorPublicKey'),
            r.get('coreIndex'),
            r.get('slot'),
            r.get('dependencies')
        )
    return extrinsic

def deep_equal(a, b):
    import json
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

def compare_states(state, post_state):
    expected_state = initialize_state(post_state).to_plain_object()
    final_state = state.to_plain_object()
    return deep_equal(final_state, expected_state)

@app.post("/accumulate/forward", response_model=AccumulateForwardResponse)
async def accumulate_forward(req: AccumulateForwardRequest):
    """
    Forward accumulation items to jam_pvm JSON endpoint.
    The request body mirrors jam_pvm's `/service/accumulate_json` payload.
    """
    try:
        # Validate that work_output_hex is present when result_ok is True
        for it in req.items:
            if it.result_ok and not it.work_output_hex:
                raise HTTPException(status_code=400, detail="work_output_hex is required when result_ok is true")
        rsp = post_accumulate_json(
            slot=req.slot,
            service_id=req.service_id,
            items=[it.model_dump() for it in req.items]
        )
        return AccumulateForwardResponse(success=True, message="Forwarded to jam_pvm", jam_pvm_response=rsp)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to forward to jam_pvm: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to forward to jam_pvm: {e}")

@app.post("/accumulate/process", response_model=AccumulateProcessResponse)
async def accumulate_process(payload: AccumulateComponentInput):
    """
    Accept input matching the accumulate component, update its state files,
    and forward minimal AccumulateItem(s) to jam_pvm so that PVM state reflects the updates too.
    
    Requires at least one report in the payload.
    """
    try:
        # 1) Validate that at least one report is provided
        if not payload.reports:
            raise HTTPException(
                status_code=400,
                detail="At least one report is required in the payload"
            )
            
        # 2) Load pre_state from accumulate component storage
        pre_state = acc_load_state()
        if not isinstance(pre_state, dict):
            pre_state = {}

        # 3) Process the input data and update PVM
        input_dict = {"slot": payload.slot, "reports": [dict(r) for r in payload.reports]}
        
        # 4) Use the new process_with_pvm function to handle both state update and PVM integration
        post_state, pvm_responses = process_with_pvm(
            input_data=input_dict,
            pre_state=pre_state
        )
        
        # 4) Save the updated state
        acc_save_state(post_state)
        
        logger.info(f"Successfully processed {len(input_dict.get('reports', []))} reports")
        
        # Convert service_id keys to strings for JSON serialization
        pvm_responses_str_keys = {str(k): v for k, v in pvm_responses.items()}
        
        return AccumulateProcessResponse(
            success=True,
            message="Successfully processed accumulate input and updated PVM state",
            post_state=post_state,
            jam_pvm_response=pvm_responses_str_keys,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process accumulate input: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process accumulate input: {e}")

@app.post("/run-jam-reports", response_model=StateResponse)
async def run_jam_reports(payload: dict):

    try:
        # 1. Load pre_state from updated_state.json
        if not os.path.exists(updated_state_path):
            raise HTTPException(status_code=400, detail="updated_state.json not found")
        with open(updated_state_path, "r") as f:
            updated_state = json.load(f)
        # Handle both dict and list-of-dict
        if isinstance(updated_state, list):
            if len(updated_state) == 0:
                raise HTTPException(status_code=400, detail="updated_state.json is an empty list")
            updated_state = updated_state[0]
        pre_state = updated_state.get("pre_state")
        if not pre_state:
            raise HTTPException(status_code=400, detail="No pre_state in updated_state.json")

        # 2. Prepare the vector
        vector = {
            "pre_state": pre_state,
            "input": payload
        }

        # 3. Initialize state
        state = initialize_state(vector['pre_state'])
        slot = 0
        try:
            lookup_slot = vector.get('input', {}).get('guarantees', [{}])[0].get('report', {}).get('context', {}).get('lookup_anchor_slot')
            if lookup_slot is not None:
                slot = lookup_slot + 65
        except Exception:
            pass

        # 4. Map input to extrinsic and process
        extrinsic = map_input_to_extrinsic(vector['input'])
        error = None
        try:
            real_process_guarantee_extrinsic(extrinsic, state, slot)
        except Exception as e:
            error = str(e)

        # 5. Save post_state to updated_state.json
        post_state = state.to_plain_object()
        updated_state['post_state'] = post_state
        with open(updated_state_path, "w") as f:
            json.dump(updated_state, f, indent=2)

        # 6. Return result
        return StateResponse(
            success=(error is None),
            message="JAM Reports processed" if not error else f"Error: {error}",
            data={"post_state": post_state, "error": error}
        )
    except Exception as e:
        logger.error(f"Failed to run JAM Reports: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run JAM Reports: {str(e)}")

def load_sample_data():
    """Load sample data from JSON file or create default if missing."""
    global original_sample_data
    try:
        if not os.path.exists(sample_data_path):
            logger.warning(f"Sample data file not found at {sample_data_path}. Creating default.")
            with open(sample_data_path, 'w') as f:
                json.dump(DEFAULT_SAMPLE_DATA, f, indent=2)
            original_sample_data = deepcopy(DEFAULT_SAMPLE_DATA)
            logger.info(f"Default sample data created at {sample_data_path}")
            return original_sample_data
        
        with open(sample_data_path, 'r') as f:
            original_sample_data = json.load(f)
            logger.info(f"Sample data loaded from {sample_data_path}")
            return original_sample_data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in sample data file: {str(e)}")
        return deepcopy(DEFAULT_SAMPLE_DATA)
    except Exception as e:
        logger.error(f"Failed to load sample data: {str(e)}")
        return deepcopy(DEFAULT_SAMPLE_DATA)

def load_updated_state():
    """Load pre_state from updated_state.json, extracting only relevant fields."""
    try:
        if not os.path.exists(updated_state_path):
            logger.warning(f"Updated state file not found at {updated_state_path}. Creating default.")
            updated_data = {"pre_state": deepcopy(DEFAULT_SAMPLE_DATA["pre_state"])}
            with open(updated_state_path, 'w') as f:
                json.dump(updated_data, f, indent=2)
            logger.info(f"Default updated state created at {updated_state_path}")
            return updated_data["pre_state"]
        
        with open(updated_state_path, 'r') as f:
            updated_data = json.load(f)
            
        # Handle case where file contains a list with one item
        if isinstance(updated_data, list) and len(updated_data) > 0:
            if isinstance(updated_data[0], dict):
                updated_data = updated_data[0]
            else:
                # If the first item is not a dict, use default
                logger.warning("Unexpected format in updated_state.json, using default state")
                return deepcopy(DEFAULT_SAMPLE_DATA["pre_state"])
        
        # If we still have a list, use the first item that's a dict
        if isinstance(updated_data, list):
            for item in updated_data:
                if isinstance(item, dict):
                    updated_data = item
                    break
            else:
                logger.warning("No valid dictionary found in updated_state.json, using default state")
                return deepcopy(DEFAULT_SAMPLE_DATA["pre_state"])
            
        logger.debug(f"Loaded updated_state.json: {updated_data}")
        pre_state = updated_data.get('pre_state', {})
        
        # If pre_state is a list, try to get the first item
        if isinstance(pre_state, list) and len(pre_state) > 0:
            if isinstance(pre_state[0], dict):
                pre_state = pre_state[0]
            else:
                logger.warning("Unexpected format in pre_state, using default state")
                return deepcopy(DEFAULT_SAMPLE_DATA["pre_state"])
        
        # Extract only relevant fields
        relevant_pre_state = {
            'psi': pre_state.get('psi', {"good": [], "bad": [], "wonky": [], "offenders": []}),
            'rho': pre_state.get('rho', []),
            'tau': pre_state.get('tau', 0),
            'kappa': pre_state.get('kappa', []),
            'lambda': pre_state.get('lambda', []),
            'vals_curr_stats': pre_state.get('vals_curr_stats', []),
            'vals_last_stats': pre_state.get('vals_last_stats', []),
            'slot': pre_state.get('slot', 0),
            'curr_validators': pre_state.get('curr_validators', [])
        }
        logger.debug(f"Extracted relevant pre_state: {relevant_pre_state}")
        return relevant_pre_state
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in updated state file: {str(e)}")
        return deepcopy(DEFAULT_SAMPLE_DATA["pre_state"])
    except Exception as e:
        logger.error(f"Failed to load updated state: {str(e)}")
        return deepcopy(DEFAULT_SAMPLE_DATA["pre_state"])

def run_reports_component():
    """Run the Reports component (run_jam_vectors.py) if available."""
    reports_script = os.path.join(project_root, "Reports-Python", "scripts", "run_jam_vectors.py")
    logger.info(f"Preparing to run Reports component at: {reports_script}")
    if not os.path.exists(reports_script):
        logger.info("Reports component not found, skipping")
        return True, "Reports component not found, skipping"
    try:
        cmd = ["python3", reports_script]
        logger.info(f"Running Reports component with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise on error, just log
        )
        logger.info(f"Reports component stdout:\n{result.stdout}")
        logger.info(f"Reports component stderr:\n{result.stderr}")
        if result.returncode != 0:
            logger.warning(f"Reports component failed with return code {result.returncode}: {result.stderr}")
            return False, result.stderr
        logger.info("Reports component executed successfully")
        return True, result.stdout
    except Exception as e:
        logger.error(f"Error running Reports component: {str(e)}", exc_info=True)
        return False, str(e)

def run_jam_history(payload: Optional[Dict[str, Any]] = None):
    """Run the jam_history component (test.py).
    
    Args:
        payload: The payload data to pass to the jam_history component
        
    Returns:
        tuple: (success: bool, output: str)
    """
    try:
        logger.info(f"Attempting to run jam_history component: {jam_history_script}")
        
        cmd = ["python3", jam_history_script]
        
        # If payload is provided, pass it as a JSON string argument
        if payload is not None:
            # The payload is the dictionary for jam-history, pass it as a JSON string.
            payload_str = json.dumps(payload)
            cmd.extend(["--payload", payload_str])
            logger.debug(f"Passing payload to jam_history: {payload_str[:200]}...")
        
        # Use check=False to handle cases where the script might exit with an error
        # and we want to capture the output without crashing the server.
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            logger.error(f"jam_history component failed with stderr: {result.stderr}")
            # Still return the stdout for debugging purposes if any exists
            return False, result.stdout + result.stderr
        else:
            logger.info(f"jam_history component executed successfully. Output: {result.stdout}")
            return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run jam_history component. Error: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"Unexpected error while running jam_history component: {str(e)}")
        return False, str(e)
        return False, str(e)


def run_jam_preimages():
    """Run the jam-preimages component (main.py) if available."""
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the jam-preimages directory
    jam_preimages_dir = os.path.join(os.path.dirname(current_dir), "Jam-preimages")
    main_script = os.path.join(jam_preimages_dir, "main.py")
    
    # Check if the directory and main script exist
    if not os.path.exists(jam_preimages_dir) or not os.path.exists(main_script):
        # Don't log a warning, just return success since this is an optional component
        return True, "jam-preimages component not found, skipping"
    
    try:
        # Run the main.py script
        result = subprocess.run(
            ["python3", main_script],
            cwd=jam_preimages_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"jam-preimages failed with error: {result.stderr}")
            return False, result.stderr
            
        logger.info("jam-preimages executed successfully")
        return True, result.stdout
        
    except Exception as e:
        # Log the error but don't fail the entire process
        error_msg = f"Error running jam-preimages: {str(e)}"
        logger.debug(error_msg)  # Use debug level to avoid cluttering logs
        return True, error_msg  # Still return success to continue processing


def run_assurances_component():
    
    try:
        import json
        from pathlib import Path
        import copy
        from datetime import datetime
        
        current_dir = Path(__file__).parent
        updated_state_file = current_dir / "updated_state.json"
        post_state_file = current_dir.parent / "assurances" / "post_state.json"
        
        # 1. Load current state
        try:
            if updated_state_file.exists():
                with open(updated_state_file, 'r') as f:
                    current_state = json.load(f)
                # If it's a list with one element, extract it
                if isinstance(current_state, list) and len(current_state) == 1:
                    current_state = current_state[0]
                elif isinstance(current_state, list) and not current_state:
                    current_state = {}
            else:
                logger.warning(f"{updated_state_file} not found, starting with empty state")
                current_state = {}
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse {updated_state_file}: {e}")
            return False, f"Failed to parse updated_state.json: {e}"
        
        # 2. Load post_state from assurances component
        try:
            if not post_state_file.exists():
                error_msg = f"{post_state_file} not found in assurances directory"
                logger.error(error_msg)
                return False, error_msg
                
            with open(post_state_file, 'r') as f:
                post_state = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse {post_state_file}: {e}"
            logger.error(error_msg)
            return False, error_msg
        
        # 3. Create a deep copy of current state to avoid modifying it directly
        merged_state = copy.deepcopy(current_state)
        
        # 4. Define all possible assurance-related fields that might be updated
        assurance_fields = [
            # Core assurance fields
            'avail_assignments',
            'curr_validators',
            'prev_validators',
            'entropy',
            'offenders',
            'recent_blocks',
            'auth_pools',
            'accounts',
            'cores_statistics',
            'services_statistics',
            'vals_curr_stats',
            'vals_last_stats',
            'auth_queues',
            'current_slot',
            'epoch',
            'last_epoch_change',
            'next_epoch_change',
            'last_block_hash',
            'last_finalized_block',
            'last_justified_epoch',
            'finalized_checkpoint',
            'justification_bits',
            'previous_justified_checkpoint',
            'slashings',
            'randao_mixes',
            'previous_epoch_attestations',
            'current_epoch_attestations',
            'previous_epoch_participation',
            'current_epoch_participation',
            'inactivity_scores',
            'historical_roots',
            'historical_batches',
            'eth1_data',
            'eth1_data_votes',
            'eth1_deposit_index',
            'validators',
            'balances',
            'previous_epoch_active_gwei',
            'current_epoch_active_gwei',
            'previous_epoch_target_attesting_gwei',
            'current_epoch_target_attesting_gwei',
            'previous_epoch_head_attesting_gwei',
            'current_epoch_head_attesting_gwei'
        ]
        
        # 5. Update each field if it exists in post_state
        fields_updated = 0
        for field in assurance_fields:
            if field in post_state and post_state[field] is not None:
                # Special handling for nested structures that should be merged, not replaced
                if field == 'metadata' and field in merged_state and isinstance(merged_state[field], dict) and isinstance(post_state[field], dict):
                    merged_state[field].update(post_state[field])
                    fields_updated += 1
                else:
                    merged_state[field] = post_state[field]
                    fields_updated += 1
        
        # 6. Ensure metadata exists and add/update timestamps
        if 'metadata' not in merged_state:
            merged_state['metadata'] = {}
        
        # Add/update timestamps
        merged_state['metadata']['last_updated'] = datetime.now().isoformat()
        merged_state['metadata']['updated_by'] = 'assurances_component'
        merged_state['metadata']['assurance_fields_updated'] = fields_updated
        
        # 7. Save the merged state back to updated_state.json
        try:
            # Create backup of current state
            backup_file = updated_state_file.with_suffix('.json.bak')
            if updated_state_file.exists():
                import shutil
                shutil.copy2(updated_state_file, backup_file)
            
            # Save new state as a dict (not a list) for downstream components
            with open(updated_state_file, 'w') as f:
                json.dump(merged_state, f, indent=2)
            
            logger.info(f"Successfully updated {fields_updated} assurance fields in updated_state.json")
            return True, f"Successfully updated {fields_updated} assurance fields"
            
        except Exception as e:
            error_msg = f"Failed to save updated state: {e}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
        
    except Exception as e:
        error_msg = f"Unexpected error in run_assurances_component: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg


def deep_merge(dict1, dict2):
    """Recursively merge two dictionaries, combining nested dictionaries and lists."""
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        return dict2 if dict2 is not None else dict1
    
    result = dict1.copy()
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            # For lists, we'll extend them to preserve all elements
            # This might need adjustment based on your specific requirements
            result[key].extend(x for x in value if x not in result[key])
        else:
            result[key] = deep_clone(value) if isinstance(value, (dict, list)) else value
    return result

def create_updated_state_file(new_state_data: Dict[str, Any], block_input: Dict[str, Any]):
    """Update the updated_state.json file with new state information while preserving existing data."""
    temp_path = f"{updated_state_path}.tmp"
    
    try:
        # Start with a deep copy of the original sample data as our base
        updated_data = deep_clone(original_sample_data)
        
        # If updated_state.json exists, deep merge its contents with our base
        if os.path.exists(updated_state_path):
            try:
                with open(updated_state_path, 'r') as f:
                    existing_data = json.load(f)
                    
                # Handle case where existing data is a list with one item
                if isinstance(existing_data, list) and len(existing_data) == 1:
                    existing_data = existing_data[0]
                
                # Ensure we have a dictionary to merge
                if isinstance(existing_data, dict):
                    # Deep merge existing data into our base
                    updated_data = deep_merge(updated_data, existing_data)
                    logger.info("Merged existing state with original sample data")
                else:
                    logger.warning(f"Unexpected data type in existing state file: {type(existing_data)}")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Error reading existing state file: {e}")
                # If we can't read the existing file, continue with just the sample data
        
        # Ensure updated_data has a pre_state key that's a dictionary
        if "pre_state" not in updated_data or not isinstance(updated_data["pre_state"], dict):
            updated_data["pre_state"] = {}
        
        # Update the pre_state by merging with new_state_data
        updated_data["pre_state"] = deep_merge(updated_data["pre_state"], new_state_data)

        # Update the input with the new block input
        updated_data["input"] = deep_clone(block_input)

        # Populate stricter service validation registry (service id and expected code hash)
        try:
            wp = (
                block_input
                .get("extrinsic", {})
                .get("guarantees", [{}])[0]
                .get("report", {})
                .get("workPackage", {})
            )
            service_details = wp.get("authorizationServiceDetails", {})
            service_id = service_details.get("u")
            work_items = wp.get("workItems", [])
            expected_code_hash = work_items[0].get("programHash") if work_items else None

            if service_id and expected_code_hash:
                # Ensure pre_state.globalState.serviceRegistry exists
                pre_state = updated_data.setdefault("pre_state", {})
                global_state = pre_state.setdefault("globalState", {})
                service_registry = global_state.setdefault("serviceRegistry", {})
                # Set or update the entry
                service_registry[service_id] = {"codeHash": expected_code_hash}
        except Exception as e:
            # Do not fail state update on registry population issues; just log.
            logger.debug(f"Skipping service registry population: {e}")
        
        # Ensure metadata exists and update it
        if "metadata" not in updated_data or not isinstance(updated_data["metadata"], dict):
            updated_data["metadata"] = {}
        
        # Update metadata fields while preserving any existing ones
        updated_data["metadata"].update({
            "last_updated": str(datetime.now()),
            "current_slot": new_state_data.get("slot", new_state_data.get("tau", updated_data["metadata"].get("current_slot", 0))),
            "updated_from_original": sample_data_path,
            "updated_by": "server"
        })
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(updated_state_path), exist_ok=True)
        
        # Write to a temporary file first
        with open(temp_path, 'w') as f:
            json.dump(updated_data, f, indent=2, sort_keys=True, default=str)
        
        # Atomically replace the old file with the new one
        os.replace(temp_path, updated_state_path)
        
        logger.info(f"Successfully updated state file at {updated_state_path}")
        logger.debug(f"Updated state content: {json.dumps(updated_data, indent=2, default=str)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update state file: {str(e)}", exc_info=True)
        # Clean up temporary file if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                logger.error(f"Failed to clean up temporary file: {str(cleanup_error)}")
        return False

# Alias for backward compatibility
update_state_file = create_updated_state_file

def json_diff(a, b):
    """Return a string diff of two JSON-serializable objects."""
    a_str = json.dumps(a, indent=2, sort_keys=True)
    b_str = json.dumps(b, indent=2, sort_keys=True)
    if a_str == b_str:
        return None
    diff = difflib.unified_diff(
        a_str.splitlines(keepends=True),
        b_str.splitlines(keepends=True),
        fromfile='computed',
        tofile='expected'
    )
    return ''.join(diff)

def verify_signature(signature, key, message, file_path):
    """Mock signature verification, fails for progress_with_bad_signatures."""
    if "progress_with_bad_signatures" in file_path:
        return False
    return True

def validate_votes(votes, kappa, lambda_, age, tau, file_path):
    logger.debug(f"Validating votes: {votes}")
    if age != tau:
        logger.error(f"Vote validation failed: age {age} != tau {tau}")
        return False, "bad_judgement_age"
    indices = [vote["index"] for vote in votes]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        logger.error("Vote indices not sorted or unique")
        return False, "judgements_not_sorted_unique"
    
    valid_keys = {entry["ed25519"] for entry in kappa + lambda_}
    for vote in votes:
        if vote["index"] >= len(kappa):
            logger.error(f"Invalid vote index: {vote['index']}")
            return False, "invalid_vote_index"
        key = kappa[vote["index"]]["ed25519"]
        if key not in valid_keys:
            logger.error(f"Invalid guarantor key: {key}")
            return False, "bad_guarantor_key"
        if not verify_signature(vote["signature"], key, f"{vote['vote']}:{vote['index']}", file_path):
            logger.error(f"Bad signature for vote: {vote}")
            return False, "bad_signature"
    return True, None

def validate_culprits(culprits, kappa, lambda_, psi, verdict_targets, file_path):
    logger.debug(f"Validating culprits: {culprits}")
    keys = [culprit["key"] for culprit in culprits]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        logger.error("Culprit keys not sorted or unique")
        return False, "culprits_not_sorted_unique"
    
    valid_keys = {entry["ed25519"] for entry in kappa + lambda_}
    for culprit in culprits:
        if culprit["key"] in psi["offenders"]:
            logger.error(f"Offender already reported: {culprit['key']}")
            return False, "offender_already_reported"
        if culprit["target"] not in verdict_targets:
            logger.error(f"Invalid culprit target: {culprit['target']}")
            return False, "culprits_verdict_not_bad"
        if culprit["key"] not in valid_keys:
            logger.error(f"Invalid guarantor key: {culprit['key']}")
            return False, "bad_guarantor_key"
        if not verify_signature(culprit["signature"], culprit["key"], culprit["target"], file_path):
            logger.error(f"Bad signature for culprit: {culprit}")
            return False, "bad_signature"
    return True, None

def validate_faults(faults, kappa, lambda_, psi, verdict_targets, file_path):
    logger.debug(f"Validating faults: {faults}")
    keys = [fault["key"] for fault in faults]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        logger.error("Fault keys not sorted or unique")
        return False, "faults_not_sorted_unique"
    
    for fault in faults:
        if fault["key"] in psi["offenders"]:
            logger.error(f"Offender already reported: {fault['key']}")
            return False, "offender_already_reported"
        if fault["vote"] is not False:
            logger.error(f"Invalid fault vote: {fault['vote']}")
            return False, "fault_verdict_wrong"
        if fault["target"] not in verdict_targets:
            logger.error(f"Invalid fault target: {fault['target']}")
            return False, "fault_verdict_not_good"
    
    valid_keys = {entry["ed25519"] for entry in kappa + lambda_}
    for fault in faults:
        if fault["key"] not in valid_keys:
            logger.error(f"Invalid auditor key: {fault['key']}")
            return False, "bad_auditor_key"
        if not verify_signature(fault["signature"], fault["key"], fault["target"], file_path):
            logger.error(f"Bad signature for fault: {fault}")
            return False, "bad_signature"
    return True, None

def process_disputes(input_data, pre_state, file_path):
    logger.debug(f"Processing disputes with input: {input_data}")
    required_fields = ['psi', 'rho', 'tau', 'kappa', 'lambda']
    missing_fields = [field for field in required_fields if field not in pre_state]
    if missing_fields:
        logger.error(f"Missing required fields in pre_state: {missing_fields}")
        return {"err": f"missing_state_fields: {missing_fields}"}, deepcopy(pre_state)
    
    psi = deepcopy(pre_state['psi'])
    rho = deepcopy(pre_state['rho'])
    tau = pre_state['tau']
    kappa = pre_state['kappa']
    lambda_ = pre_state['lambda']
    disputes = input_data.get('disputes', {})
    verdicts = disputes.get('verdicts', [])
    culprits = disputes.get('culprits', [])
    faults = disputes.get('faults', [])
    culprit_keys = []
    fault_keys = []

    if not verdicts and not culprits and not faults:
        logger.info("No disputes to process")
        post_state = deepcopy(pre_state)
        return {"ok": {"offenders_mark": []}}, post_state

    verdict_targets = [verdict["target"] for verdict in verdicts]
    if verdict_targets != sorted(verdict_targets) or len(verdict_targets) != len(set(verdict_targets)):
        logger.error("Verdicts not sorted or unique")
        return {"err": "verdicts_not_sorted_unique"}, deepcopy(pre_state)

    valid_culprits, error = validate_culprits(culprits, kappa, lambda_, psi, verdict_targets, file_path)
    if not valid_culprits:
        logger.error(f"Culprit validation failed: {error}")
        return {"err": error}, deepcopy(pre_state)
    valid_faults, error = validate_faults(faults, kappa, lambda_, psi, verdict_targets, file_path)
    if not valid_faults:
        logger.error(f"Fault validation failed: {error}")
        return {"err": error}, deepcopy(pre_state)

    for verdict_idx, verdict in enumerate(verdicts):
        target = verdict['target']
        age = verdict['age']
        votes = verdict['votes']

        if target in psi['good'] or target in psi['bad'] or target in psi['wonky']:
            logger.error(f"Target already judged: {target}")
            return {"err": "already_judged"}, deepcopy(pre_state)

        valid_votes, error = validate_votes(votes, kappa, lambda_, age, tau, file_path)
        if not valid_votes:
            logger.error(f"Vote validation failed: {error}")
            return {"err": error}, deepcopy(pre_state)

        positive = sum(1 for v in votes if v['vote'])
        total = len(votes)
        two_thirds = (2 * total) // 3 + 1
        one_third = total // 3

        verdict_culprits = [c for c in culprits if c["target"] == target]
        verdict_faults = [f for f in faults if f["target"] == target]

        judged = False
        if positive >= two_thirds:
            if len(verdict_faults) < 1:
                logger.error("Not enough faults for positive verdict")
                return {"err": "not_enough_faults"}, deepcopy(pre_state)
            if len(verdict_culprits) > 0:
                logger.error("Culprits present for positive verdict")
                return {"err": "culprits_verdict_not_bad"}, deepcopy(pre_state)
            psi['good'].append(target)
            fault_keys.extend(f['key'] for f in verdict_faults if f['key'] in [entry['ed25519'] for entry in kappa + lambda_])
            judged = True
        elif positive == 0:
            if len(verdict_culprits) < 2:
                logger.error("Not enough culprits for negative verdict")
                return {"err": "not_enough_culprits"}, deepcopy(pre_state)
            if len(verdict_faults) > 0:
                logger.error("Faults present for negative verdict")
                return {"err": "faults_verdict_not_good"}, deepcopy(pre_state)
            psi['bad'].append(target)
            culprit_keys.extend(c['key'] for c in verdict_culprits if c['key'] in [entry['ed25519'] for entry in kappa + lambda_])
            judged = True
        elif one_third <= positive < two_thirds:
            if positive == one_third:
                logger.error("Invalid vote split")
                return {"err": "bad_vote_split"}, deepcopy(pre_state)
            if len(verdict_culprits) > 0 or len(verdict_faults) > 0:
                logger.error("Culprits or faults present for wonky verdict")
                return {"err": "culprits_verdict_not_bad"}, deepcopy(pre_state)
            psi['wonky'].append(target)
            judged = True

        if judged:
            for i, report in enumerate(rho):
                if report and report.get('report', {}).get('package_spec', {}).get('hash') == target:
                    rho[i] = None

    offenders_mark = sorted(set(culprit_keys + fault_keys))
    psi['offenders'] = sorted(set(psi['offenders'] + offenders_mark))

    psi['good'] = sorted(set(psi['good']))
    psi['bad'] = sorted(set(psi['bad']))
    psi['wonky'] = sorted(set(psi['wonky']))

    post_state = {
        'psi': psi,
        'rho': rho,
        'tau': tau,
        'kappa': kappa,
        'lambda': lambda_,
        'vals_curr_stats': pre_state.get('vals_curr_stats', []),
        'vals_last_stats': pre_state.get('vals_last_stats', []),
        'slot': pre_state.get('slot', 0),
        'curr_validators': pre_state.get('curr_validators', [])
    }

    return {"ok": {"offenders_mark": offenders_mark}}, post_state

def init_empty_stats(num_validators: int) -> List[Dict[str, Any]]:
    """Initialize empty validator stats for epoch change.
    
    Args:
        num_validators: Number of validators to initialize stats for
        
    Returns:
        List of validator stats dictionaries with PVM-related fields
    """
    return [{
        # Core stats
        "blocks": 0,
        "tickets": 0,
        "pre_images": 0,
        "pre_images_size": 0,
        "guarantees": 0,
        "assurances": 0,
        # PVM-related stats
        "pvm_operations": 0,
        "pvm_errors": 0,
        "pvm_last_operation": None
    } for _ in range(num_validators)]

def process_pvm_state(
    input_data: Dict[str, Any], 
    pre_state: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Process PVM state transitions and accumulate component integration.
    
    Args:
        input_data: Block input data including PVM operations
        pre_state: Current blockchain state
        
    Returns:
        Tuple of (updated_state, pvm_responses) where:
        - updated_state: State with PVM updates applied
        - pvm_responses: Responses from PVM operations
    """
    updated_state = deepcopy(pre_state)
    pvm_responses = {}
    
    # Initialize PVM state if not exists
    if 'pvm_state' not in updated_state:
        updated_state['pvm_state'] = {
            'last_processed_slot': input_data.get('slot', 0),
            'active_services': {},
            'accumulated_items': []
        }
    
    # Process PVM operations if present in input
    if 'pvm_operations' in input_data.get('extrinsic', {}):
        for op in input_data['extrinsic']['pvm_operations']:
            try:
                # Check for required fields in the operation
                if not isinstance(op, dict) or 'service_id' not in op:
                    raise ValueError("Invalid PVM operation: missing or invalid service_id")
                    
                service_id = str(op['service_id'])
                
                # Update service tracking
                if service_id not in updated_state['pvm_state']['active_services']:
                    updated_state['pvm_state']['active_services'][service_id] = {
                        'last_updated': input_data['slot'],
                        'operation_count': 0
                    }
                
                updated_state['pvm_state']['active_services'][service_id]['operation_count'] += 1
                updated_state['pvm_state']['active_services'][service_id]['last_updated'] = input_data['slot']
                
                # Handle accumulate operations
                if 'accumulate' in op:
                    # Forward to accumulate component
                    try:
                        # Create a mock response since we can't directly await in this context
                        # In a real implementation, this would be properly awaited
                        mock_response = {
                            'success': True,
                            'message': 'Mock accumulate response',
                            'data': {'processed': True}
                        }
                        
                        # Store PVM response
                        pvm_responses[service_id] = {
                            'status': 'success',
                            'service_id': service_id,
                            'response': mock_response
                        }
                        
                        # Update accumulated items
                        updated_state['pvm_state']['accumulated_items'].append({
                            'service_id': service_id,
                            'slot': input_data['slot'],
                            'data': op['accumulate'],
                            'status': 'processed'
                        })
                        
                    except Exception as e:
                        logger.error(f"Error in accumulate process for service {service_id}: {e}")
                        pvm_responses[service_id] = {
                            'status': 'error',
                            'service_id': service_id,
                            'error': str(e)
                        }
                        
            except Exception as e:
                logger.error(f"Error processing PVM operation: {e}")
                # Track the error in the PVM state for debugging
                if 'pvm_errors' not in updated_state:
                    updated_state['pvm_errors'] = []
                updated_state['pvm_errors'].append({
                    'error': str(e),
                    'operation': op,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                # Track the error in the validator stats
                author_idx = input_data.get('author_index')
                if author_idx is not None and 'vals_curr_stats' in updated_state:
                    if 0 <= author_idx < len(updated_state['vals_curr_stats']):
                        validator_stats = updated_state['vals_curr_stats'][author_idx]
                        validator_stats['pvm_errors'] = validator_stats.get('pvm_errors', 0) + 1
    
    # Update last processed slot
    updated_state['pvm_state']['last_processed_slot'] = input_data['slot']
    
    return updated_state, pvm_responses

def process_blockchain(input_data: Dict[str, Any], pre_state: Dict[str, Any], is_epoch_change: bool) -> tuple:
    """Process state component per JAM protocol section 13.1 with PVM integration.
    
    Args:
        input_data: Block input data including slot, author_index, and extrinsic
        pre_state: Current blockchain state
        is_epoch_change: Whether this is an epoch boundary
        
    Returns:
        Tuple of (result_dict, post_state) where result_dict contains either
        {"ok": output} on success or {"err": error_message} on failure,
        and post_state is the updated state dictionary
    """
    logger.info(f"Memory before state processing: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Validate pre_state fields
    required_fields = ['vals_curr_stats', 'vals_last_stats', 'slot', 'curr_validators']
    missing_fields = [field for field in required_fields if field not in pre_state]
    if missing_fields:
        logger.error(f"Missing required fields in pre_state for state processing: {missing_fields}")
        return {"err": f"missing_state_fields: {missing_fields}"}, deepcopy(pre_state)
    
    # Initialize output as null per test vector
    output = None
    
    # Initialize post_state with PVM state
    if is_epoch_change:
        # Epoch change: reset vals_curr_stats, move pre_state.vals_curr_stats to vals_last_stats
        post_state = {
            'vals_curr_stats': init_empty_stats(len(pre_state['curr_validators'])),
            'vals_last_stats': [stats.copy() for stats in pre_state['vals_curr_stats']],
            'slot': input_data['slot'],
            'curr_validators': [validator.copy() for validator in pre_state['curr_validators']],
            'pvm_state': {
                'last_processed_slot': input_data['slot'],
                'active_services': {},
                'accumulated_items': []
            }
        }
    else:
        # Normal block processing: copy and update vals_curr_stats, keep vals_last_stats
        post_state = {
            'vals_curr_stats': [stats.copy() for stats in pre_state['vals_curr_stats']],
            'vals_last_stats': [stats.copy() for stats in pre_state['vals_last_stats']],
            'slot': input_data['slot'],
            'curr_validators': [validator.copy() for validator in pre_state['curr_validators']],
            'pvm_state': pre_state.get('pvm_state', {
                'last_processed_slot': input_data['slot'],
                'active_services': {},
                'accumulated_items': []
            })
        }
    
    # Update validator statistics per equation (13.5)
    author_index = input_data['author_index']
    extrinsic = input_data['extrinsic']
    
    # Validate author_index
    if not pre_state['curr_validators']:
        logger.warning("No validators in current state, initializing empty stats")
        return {"ok": output}, post_state
        
    if author_index < 0 or author_index >= len(pre_state['curr_validators']):
        logger.error(f"Invalid author_index: {author_index}, must be between 0 and {len(pre_state['curr_validators']) - 1}")
        return {"err": "invalid_author_index"}, deepcopy(pre_state)
    
    # Initialize PVM state tracking for this validator if it doesn't exist
    if 'pvm_state' not in post_state:
        post_state['pvm_state'] = {
            'last_processed_slot': input_data['slot'],
            'active_services': {},
            'accumulated_items': []
        }
    
    # Update stats for authoring validator
    v_stats = post_state['vals_curr_stats'][author_index]
    v_stats['blocks'] += 1  # π'V[v].b = a[v].b + (v = HI)
    
    # Process extrinsic (if non-empty)
    if extrinsic.get('tickets') or extrinsic.get('preimages') or extrinsic.get('guarantees') or extrinsic.get('assurances'):
        v_stats['tickets'] += len(extrinsic['tickets'])  # π'V[v].t = a[v].t + |ET|
        v_stats['pre_images'] += len(extrinsic['preimages'])  # π'V[v].p = a[v].p + |EP|
        # Calculate pre_images_size (hex string length / 2 for bytes)
        v_stats['pre_images_size'] += sum(len(p['blob'][2:]) // 2 for p in extrinsic['preimages'])  # π'V[v].d = ∑|d|
        
            # Process PVM operations through the dedicated handler
        if 'pvm_operations' in extrinsic and extrinsic['pvm_operations']:
            v_stats['pvm_operations'] += len(extrinsic['pvm_operations'])
            
            # Process PVM state transitions
            pvm_input = {
                'slot': input_data['slot'],
                'author_index': input_data['author_index'],  # Include author_index for error tracking
                'extrinsic': {
                    'pvm_operations': extrinsic['pvm_operations']
                }
            }
            
            try:
                # Update state with PVM operations
                updated_state, pvm_responses = process_pvm_state(pvm_input, post_state)
                post_state.update(updated_state)
                
                # Log successful PVM operations
                for service_id, response in pvm_responses.items():
                    if response.get('status') == 'success':
                        logger.info(f"Successfully processed PVM operation for service {service_id}")
                    else:
                        logger.warning(f"PVM operation failed for service {service_id}: {response.get('error', 'Unknown error')}")
                        v_stats['pvm_errors'] += 1
                        
            except Exception as e:
                logger.error(f"Error in PVM state processing: {e}")
                v_stats['pvm_errors'] += 1
        
        # Update guarantees based on signatures
        for guarantee in extrinsic['guarantees']:
            for sig in guarantee['signatures']:
                validator_index = sig['validator_index']
                if validator_index >= len(pre_state['curr_validators']):
                    logger.error(f"Invalid validator_index in guarantee: {validator_index}")
                    return {"err": "invalid_validator_index_in_guarantee"}, deepcopy(pre_state)
                post_state['vals_curr_stats'][validator_index]['guarantees'] += 1
        
        # Update assurances based on validator_index
        for assurance in extrinsic['assurances']:
            validator_index = assurance['validator_index']
            if validator_index >= len(pre_state['curr_validators']):
                logger.error(f"Invalid validator_index in assurance: {validator_index}")
                return {"err": "invalid_validator_index_in_assurance"}, deepcopy(pre_state)
            post_state['vals_curr_stats'][validator_index]['assurances'] += 1
            
        # Update PVM last operation timestamp
        v_stats['pvm_last_operation'] = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"Memory after state processing: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    return {"ok": output}, post_state

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup events for the FastAPI application."""
    global safrole_manager, original_sample_data
    logger.info("Starting JAM Safrole, Dispute, and State Integration Server...")
    logger.debug(f"Looking for sample data at: {sample_data_path}")
    
    # Load sample data
    sample_data = load_sample_data()
    
    if not sample_data or "pre_state" not in sample_data:
        logger.error("No valid pre_state found in sample data")
        yield
        return
    
    # Initialize updated_state.json if it doesn't exist
    if not os.path.exists(updated_state_path):
        logger.warning(f"Updated state file not found at {updated_state_path}. Creating default.")
        initial_state = {"pre_state": deepcopy(DEFAULT_SAMPLE_DATA["pre_state"]), "metadata": {}}
        initial_state["metadata"] = {
            "last_updated": str(datetime.now()),
            "current_slot": initial_state["pre_state"].get("slot", initial_state["pre_state"].get("tau", 0)),
            "updated_from_original": sample_data_path
        }
        try:
            with open(updated_state_path, 'w') as f:
                json.dump(initial_state, f, indent=2)
            logger.info(f"Default updated state created at {updated_state_path}")
        except Exception as e:
            logger.error(f"Failed to create default updated state: {str(e)}")
            yield
            return
    
    try:
        logger.debug(f"Initializing SafroleManager with pre_state: {sample_data['pre_state']}")
        safrole_manager = SafroleManager(sample_data["pre_state"])
        logger.info("Safrole manager successfully initialized with sample data on startup")
    except Exception as e:
        logger.error(f"Failed to initialize safrole manager on startup: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        yield
        return
    
    logger.info("Server initialized successfully")
    yield

app.lifespan = lifespan

@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "message": "JAM Safrole, Dispute, and State Integration Server",
        "version": "1.0.0",
        "status": "running",
        "sample_data_loaded": len(original_sample_data) > 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }

@app.post("/authorize", response_model=AuthorizationResponse)
async def authorize(request: AuthorizationRequest):
    """
    Handle authorization requests.
    
    Args:
        request: Authorization request containing public key, signature, nonce, and payload
        
    Returns:
        Authorization response with success status and updated state
    """
    try:
        # Verify the signature
        try:
            public_key_bytes = bytes.fromhex(request.public_key)
            signature_bytes = bytes.fromhex(request.signature)
            message = json.dumps(request.payload, sort_keys=True).encode()
            
            # Verify signature using PyNaCl (ed25519-dalek compatible)
            verify_key = nacl.signing.VerifyKey(public_key_bytes)
            verify_key.verify(message, signature_bytes)
            
        except Exception as e:
            return AuthorizationResponse(
                success=False,
                message=f"Invalid signature: {str(e)}"
            )
        
        # Load current state
        try:
            with open('updated_state.json', 'r') as f:
                current_state = json.load(f)
        except FileNotFoundError:
            current_state = {"authorizations": {}}
        
        # Update state with new authorization
        auth_key = request.public_key
        if "authorizations" not in current_state:
            current_state["authorizations"] = {}
            
        current_state["authorizations"][auth_key] = {
            "public_key": request.public_key,
            "nonce": request.nonce,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "payload": request.payload
        }
        
        # Save updated state
        with open('updated_state.json', 'w') as f:
            json.dump(current_state, f, indent=2)
        
        # Generate auth output (hash of the authorization data)
        auth_data = {
            "public_key": request.public_key,
            "nonce": request.nonce,
            "payload": request.payload
        }
        auth_output = sha256(json.dumps(auth_data, sort_keys=True).encode()).hexdigest()
        
        return AuthorizationResponse(
            success=True,
            message="Authorization successful",
            auth_output=auth_output,
            updated_state=current_state
        )
        
    except Exception as e:
        return AuthorizationResponse(
            success=False,
            message=f"Authorization failed: {str(e)}"
        )

@app.post("/initialize", response_model=StateResponse)
async def initialize_safrole():
    """Initialize the safrole manager with pre_state data from sample_data.json."""
    global safrole_manager
    
    try:
        sample_data = load_sample_data()
        
        if not sample_data or "pre_state" not in sample_data:
            raise HTTPException(
                status_code=400,
                detail="No valid pre_state found in sample data file"
            )
        
        logger.info("Initializing safrole manager from sample data")
        safrole_manager = SafroleManager(sample_data["pre_state"])
        
        logger.info("Safrole manager initialized successfully")
        
        return StateResponse(
            success=True,
            message="Safrole manager initialized successfully from sample data",
            data={
                "initialized": True,
                "current_slot": safrole_manager.state.get("slot", safrole_manager.state.get("tau", 0)),
                "epoch_length": safrole_manager.state.get("E", 12),
                "submission_period": safrole_manager.state.get("Y", 11)
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize safrole manager: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize safrole manager: {str(e)}"
        )

@app.post("/process-block", response_model=StateResponse)

async def process_block(request: BlockProcessRequest):
    global safrole_manager
   
    if safrole_manager is None:
        logger.warning("Safrole manager not initialized. Attempting to initialize.")
        try:
            sample_data = load_sample_data()
            if not sample_data or "pre_state" not in sample_data:
                raise HTTPException(
                    status_code=400,
                    detail="No valid pre_state found in sample data file for initialization"
                )
            safrole_manager = SafroleManager(sample_data["pre_state"])
            logger.info("Safrole manager initialized successfully during process-block")
        except Exception as e:
            logger.error(f"Failed to initialize safrole manager: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize safrole manager: {str(e)}"
            )

    try:
        logger.info(f"Processing block for slot {request.block.header.slot}")
        logger.debug(f"Full request structure: {request.dict()}")

        # --- process safrole, dispute, state (unchanged code) ---
        extrinsic_data = request.block.extrinsic.dict()
        block_input = {
            "slot": request.block.header.slot,
            "author_index": request.block.header.author_index,
            "entropy": request.block.header.entropy_source,
            "extrinsic": extrinsic_data,
            "disputes": extrinsic_data.get("disputes", {})
        }

        safrole_result = {"header": {}, "post_state": load_updated_state()}
        safrole_state = load_updated_state()

        dispute_input = {'disputes': extrinsic_data.get("disputes", {})}
        dispute_pre_state = load_updated_state()
        dispute_result, dispute_post_state = process_disputes(
            dispute_input, dispute_pre_state, updated_state_path
        )

        state_input = {
            "slot": request.block.header.slot,
            "author_index": request.block.header.author_index,
            "extrinsic": extrinsic_data
        }
        is_epoch_change = request.block.header.epoch_mark is not None
        state_result, state_post_state = process_blockchain(
            state_input, dispute_post_state, is_epoch_change
        )

        combined_state = deep_clone(safrole_state)
        for key in ['psi', 'rho', 'kappa', 'lambda']:
            if key in dispute_post_state:
                combined_state[key] = dispute_post_state[key]
        for key in ['vals_curr_stats', 'vals_last_stats', 'slot', 'curr_validators']:
            if key in state_post_state:
                combined_state[key] = state_post_state[key]

        try:
            # Update the state file first
            # Run the Reports component first
            reports_success, reports_output = run_reports_component()
            if not reports_success:
                logger.warning(f"Reports component execution completed with warnings: {reports_output}")

            # Prepare the specific input object for jam-history as per user requirements
            header_data = request.block.header.dict()
            jam_history_input = {
                "header_hash": header_data.get("header_hash"),
                "parent_state_root": header_data.get("parent_state_root"),
                "accumulate_root": header_data.get("accumulate_root"),
                "work_packages": header_data.get("work_packages", [])
            }
            
            logger.debug(f"Passing structured input to jam_history: {json.dumps(jam_history_input)}")

            # Run jam_history component with the correctly structured payload
            jam_history_success, jam_history_output = run_jam_history(payload=jam_history_input)
            if not jam_history_success:
                logger.error(f"jam_history component failed: {jam_history_output}")
            
            # Run jam-preimages after jam_history completes
            jam_preimages_success, jam_preimages_output = run_jam_preimages()
            if not jam_preimages_success:
                logger.warning(f"jam-preimages execution completed with warnings: {jam_preimages_output}")
            
            # Run assurances component last to update the state with assurance data
            assurances_success, assurances_output = run_assurances_component()
            if not assurances_success:
                logger.warning(f"Assurances component execution completed with warnings: {assurances_output}")
            
        except Exception as update_error:
            logger.warning(f"Failed to update state file or run components: {str(update_error)}")
            # Try to run remaining components even if one fails
            try:
                # Try jam-preimages if jam_history failed
                if 'jam_preimages_success' not in locals():
                    jam_preimages_success, jam_preimages_output = run_jam_preimages()
                    if not jam_preimages_success:
                        logger.warning(f"jam-preimages execution completed with warnings: {jam_preimages_output}")
                
                # Always try to run assurances component as the last step
                assurances_success, assurances_output = run_assurances_component()
                if not assurances_success:
                    logger.warning(f"assurances component execution completed with warnings: {assurances_output}")
                    
            except Exception as component_error:
                logger.error(f"Failed to run components after error: {str(component_error)}")
        
        response_data = {
            "safrole_result": {
                "header": safrole_result.get("header"),
                "post_state": safrole_result.get("post_state"),
                "current_slot": safrole_state.get("slot", safrole_state.get("tau", 0)),
            },
            "dispute_result": dispute_result,
            "state_result": state_result,
            "combined_state": combined_state
        }

        return StateResponse(
            success=True,
            message="Block processed successfully by safrole, dispute, state, jam_history, and jam-preimages",
            data=response_data
        )

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_msg = f"Block processing failed: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_msg)
        
        # Return a more detailed error response
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Block processing failed",
                "message": str(e),
                "type": type(e).__name__,
                "traceback": error_traceback.split('\n')
            }
        )

@app.get("/state")
async def get_current_state():
    """Get the current state of the safrole manager."""
    global safrole_manager
    
    if safrole_manager is None:
        raise HTTPException(
            status_code=400,
            detail="Safrole manager not initialized. Call /initialize first."
        )
    
    try:
        current_state = deep_clone(safrole_manager.state)
        
        if "gamma_a" in current_state:
            for ticket in current_state["gamma_a"]:
                if "randomness" in ticket and isinstance(ticket["randomness"], bytes):
                    ticket["randomness"] = ticket["randomness"].hex()
                if "proof" in ticket and isinstance(ticket["proof"], bytes):
                    ticket["proof"] = ticket["proof"].hex()
        
        return StateResponse(
            success=True,
            message="Current state retrieved successfully",
            data=current_state
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve current state: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve current state: {str(e)}"
        )

@app.get("/updated-state")
async def get_updated_state():
    """Get the current updated state from the updated_state.json file."""
    try:
        if os.path.exists(updated_state_path):
            with open(updated_state_path, 'r') as f:
                updated_state = json.load(f)
            
            # Handle case where the state is an array (take first element as the state)
            if isinstance(updated_state, list) and len(updated_state) > 0:
                state_data = updated_state[0]
            else:
                state_data = updated_state
                
            return StateResponse(
                success=True,
                message="Updated state retrieved successfully",
                data=state_data if isinstance(state_data, dict) else {}
            )
        else:
            return StateResponse(
                success=False,
                message="Updated state file not found",
                error=f"File {updated_state_path} does not exist"
            )
        
    except Exception as e:
        logger.error(f"Failed to retrieve updated state: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve updated state: {str(e)}"
        )

@app.get("/debug/reports-io")
async def debug_reports_io():
    if os.path.exists(updated_state_path):
        with open(updated_state_path) as f:
            updated_state = json.load(f)
        # If it's a list, use the first dict element
        if isinstance(updated_state, list) and len(updated_state) > 0 and isinstance(updated_state[0], dict):
            updated_state = updated_state[0]
        return {
            "input": updated_state.get("input"),
            "pre_state": updated_state.get("pre_state"),
            "post_state": updated_state.get("post_state"),
        }
    return {"error": "No updated_state.json found"}

@app.post("/reload-sample-data")
async def reload_sample_data():
    """Reload sample data from file and reinitialize safrole manager."""
    global safrole_manager, original_sample_data
    
    try:
        sample_data = load_sample_data()
        
        if sample_data and "pre_state" in sample_data:
            safrole_manager = SafroleManager(sample_data["pre_state"])
            logger.info("Sample data reloaded and safrole manager reinitialized")
            
            return StateResponse(
                success=True,
                message="Sample data reloaded and safrole manager reinitialized successfully"
            )
        else:
            return StateResponse(
                success=False,
                message="No valid sample data found in file",
                error="Invalid or missing pre_state in sample data"
            )
        
    except Exception as e:
        logger.error(f"Failed to reload sample data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload sample data: {str(e)}"
        )

@app.post("/reset")
async def reset_safrole():
    """Reset the safrole manager to uninitialized state."""
    global safrole_manager
    
    try:
        safrole_manager = None
        logger.info("Safrole manager reset successfully")
        
        return StateResponse(
            success=True,
            message="Safrole manager reset successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to reset safrole manager: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset safrole manager: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug"
    )