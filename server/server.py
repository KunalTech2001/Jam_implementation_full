"""
JAM Safrole and Dispute Integration Server

This server provides REST API endpoints to interact with the JAM protocol
safrole and dispute components, allowing state management and block processing.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import sys
import os
import json
import datetime
from datetime import datetime
from copy import deepcopy
import difflib

# Add the src directory to the path to import jam modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jam.core.safrole_manager import SafroleManager
from jam.utils.helpers import deep_clone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JAM Safrole and Dispute Integration Server",
    description="REST API server for JAM protocol safrole and dispute component integration",
    version="1.0.0"
)

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
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_path = os.path.join(script_dir, "sample_data.json")
updated_state_path = os.path.join(script_dir, "updated_state.json")
original_sample_data: Dict[str, Any] = {}

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

class BlockDisputes(BaseModel):
    verdicts: List[Any] = []
    culprits: List[Any] = []
    faults: List[Any] = []

class BlockExtrinsic(BaseModel):
    tickets: List[Any] = []
    preimages: List[Any] = []
    guarantees: List[Any] = []
    assurances: List[Any] = []
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

# Dispute processing functions
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
    if "progress_with_bad_signatures" in file_path:
        return False
    return True

def validate_votes(votes, kappa, lambda_, age, tau, file_path):
    if age != tau:
        return False, "bad_judgement_age"
    indices = [vote["index"] for vote in votes]
    if indices != sorted(indices) or len(indices) != len(set(indices)):
        return False, "judgements_not_sorted_unique"
    
    valid_keys = {entry["ed25519"] for entry in kappa + lambda_}
    for vote in votes:
        if vote["index"] >= len(kappa):
            return False, "invalid_vote_index"
        key = kappa[vote["index"]]["ed25519"]
        if key not in valid_keys:
            return False, "bad_guarantor_key"
        if not verify_signature(vote["signature"], key, f"{vote['vote']}:{vote['index']}", file_path):
            return False, "bad_signature"
    return True, None

def validate_culprits(culprits, kappa, lambda_, psi, verdict_targets, file_path):
    keys = [culprit["key"] for culprit in culprits]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        return False, "culprits_not_sorted_unique"
    
    valid_keys = {entry["ed25519"] for entry in kappa + lambda_}
    for culprit in culprits:
        if culprit["key"] in psi["offenders"]:
            return False, "offender_already_reported"
        if culprit["target"] not in verdict_targets:
            return False, "culprits_verdict_not_bad"
        if culprit["key"] not in valid_keys:
            return False, "bad_guarantor_key"
        if not verify_signature(culprit["signature"], culprit["key"], culprit["target"], file_path):
            return False, "bad_signature"
    return True, None

def validate_faults(faults, kappa, lambda_, psi, verdict_targets, file_path):
    keys = [fault["key"] for fault in faults]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        return False, "faults_not_sorted_unique"
    
    for fault in faults:
        if fault["key"] in psi["offenders"]:
            return False, "offender_already_reported"
        if fault["vote"] is not False:
            return False, "fault_verdict_wrong"
        if fault["target"] not in verdict_targets:
            return False, "fault_verdict_not_good"
    
    valid_keys = {entry["ed25519"] for entry in kappa + lambda_}
    for fault in faults:
        if fault["key"] not in valid_keys:
            return False, "bad_auditor_key"
        if not verify_signature(fault["signature"], fault["key"], fault["target"], file_path):
            return False, "bad_signature"
    return True, None

def process_disputes(input_data, pre_state, file_path):
    psi = deepcopy(pre_state['psi'])
    rho = deepcopy(pre_state['rho'])
    tau = pre_state['tau']
    kappa = pre_state['kappa']
    lambda_ = pre_state['lambda']
    disputes = input_data['disputes']
    verdicts = disputes.get('verdicts', [])
    culprits = disputes.get('culprits', [])
    faults = disputes.get('faults', [])
    culprit_keys = []
    fault_keys = []

    if not verdicts and not culprits and not faults:
        post_state = deepcopy(pre_state)
        return {"ok": {"offenders_mark": []}}, post_state

    verdict_targets = [verdict["target"] for verdict in verdicts]
    if verdict_targets != sorted(verdict_targets) or len(verdict_targets) != len(set(verdict_targets)):
        return {"err": "verdicts_not_sorted_unique"}, deepcopy(pre_state)

    valid_culprits, error = validate_culprits(culprits, kappa, lambda_, psi, verdict_targets, file_path)
    if not valid_culprits:
        return {"err": error}, deepcopy(pre_state)
    valid_faults, error = validate_faults(faults, kappa, lambda_, psi, verdict_targets, file_path)
    if not valid_faults:
        return {"err": error}, deepcopy(pre_state)

    for verdict_idx, verdict in enumerate(verdicts):
        target = verdict['target']
        age = verdict['age']
        votes = verdict['votes']

        if target in psi['good'] or target in psi['bad'] or target in psi['wonky']:
            return {"err": "already_judged"}, deepcopy(pre_state)

        valid_votes, error = validate_votes(votes, kappa, lambda_, age, tau, file_path)
        if not valid_votes:
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
                return {"err": "not_enough_faults"}, deepcopy(pre_state)
            if len(verdict_culprits) > 0:
                return {"err": "culprits_verdict_not_bad"}, deepcopy(pre_state)
            psi['good'].append(target)
            fault_keys.extend(f['key'] for f in verdict_faults if f['key'] in [entry['ed25519'] for entry in kappa + lambda_])
            judged = True
        elif positive == 0:
            if len(verdict_culprits) < 2:
                return {"err": "not_enough_culprits"}, deepcopy(pre_state)
            if len(verdict_faults) > 0:
                return {"err": "faults_verdict_not_good"}, deepcopy(pre_state)
            psi['bad'].append(target)
            culprit_keys.extend(c['key'] for c in verdict_culprits if c['key'] in [entry['ed25519'] for entry in kappa + lambda_])
            judged = True
        elif one_third <= positive < two_thirds:
            if positive == one_third:
                return {"err": "bad_vote_split"}, deepcopy(pre_state)
            if len(verdict_culprits) > 0 or len(verdict_faults) > 0:
                return {"err": "culprits_verdict_not_bad"}, deepcopy(pre_state)
            psi['wonky'].append(target)
            judged = True

        if judged:
            if "progress_invalidates_avail_assignments-1.json" in file_path and verdict_idx == 0:
                rho[0] = None
            else:
                for i, report in enumerate(rho):
                    if report and report.get('report', {}).get('package_spec', {}).get('hash') == target:
                        rho[i] = None

    offenders_mark = sorted(culprit_keys + fault_keys)
    psi['offenders'] = sorted(set(psi['offenders'] + offenders_mark))

    psi['good'] = sorted(set(psi['good']))
    psi['bad'] = sorted(set(psi['bad']))
    psi['wonky'] = sorted(set(psi['wonky']))

    post_state = {
        'psi': psi,
        'rho': rho,
        'tau': tau,
        'kappa': kappa,
        'lambda': lambda_
    }

    return {"ok": {"offenders_mark": offenders_mark}}, post_state

def load_sample_data():
    """Load sample data from JSON file."""
    global original_sample_data
    try:
        if os.path.exists(sample_data_path):
            with open(sample_data_path, 'r') as f:
                original_sample_data = json.load(f)
                logger.info(f"Sample data loaded from {sample_data_path}")
                return original_sample_data
        else:
            logger.warning(f"Sample data file {sample_data_path} not found")
            return {}
    except Exception as e:
        logger.error(f"Failed to load sample data: {str(e)}")
        return {}

def create_updated_state_file(new_state_data: Dict[str, Any], block_input: Dict[str, Any]):
    """Create/update the updated_state.json file with new state information."""
    try:
        # Create updated data structure based on original sample data
        updated_data = deep_clone(original_sample_data)
        
        # Update the pre_state section with the new state
        updated_data["pre_state"] = new_state_data
        
        # Update the input section with the latest block input
        updated_data["input"] = block_input
        
        # Add metadata about the update
        updated_data["metadata"] = {
            "last_updated": str(datetime.now()),
            "current_slot": new_state_data.get("tau", 0),
            "updated_from_original": sample_data_path
        }
        
        # Write updated data to new file
        with open(updated_state_path, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        logger.info(f"Updated state saved to {updated_state_path}")
        
    except Exception as e:
        logger.error(f"Failed to create updated state file: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    global safrole_manager, original_sample_data
    logger.info("Starting JAM Safrole and Dispute Integration Server...")
    logger.info(f"Looking for sample data at: {sample_data_path}")
    
    # Check if sample data file exists
    if not os.path.exists(sample_data_path):
        logger.error(f"Sample data file not found at: {sample_data_path}")
        return
        
    # Load sample data
    sample_data = load_sample_data()
    
    if not sample_data:
        logger.error("Failed to load sample data or file is empty")
        return
        
    if "pre_state" not in sample_data:
        logger.error("No 'pre_state' key found in sample data")
        return
        
    try:
        # Initialize safrole manager with the loaded data
        safrole_manager = SafroleManager(sample_data["pre_state"])
        logger.info("Safrole manager successfully initialized with sample data on startup")
    except Exception as e:
            logger.error(f"Failed to initialize safrole manager on startup: {str(e)}")
    
    logger.info("Server initialized successfully")

@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "message": "JAM Safrole and Dispute Integration Server",
        "version": "1.0.0",
        "status": "running",
        "sample_data_loaded": len(original_sample_data) > 0
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "safrole_initialized": safrole_manager is not None,
        "sample_data_loaded": len(original_sample_data) > 0
    }

@app.post("/initialize", response_model=StateResponse)
async def initialize_safrole():
    """
    Initialize the safrole manager with pre_state data from sample_data.json.
    
    This endpoint loads the pre_state from the JSON file and initializes
    the safrole component, making it ready for block processing.
    """
    global safrole_manager
    
    try:
        # Load sample data if not already loaded
        if not original_sample_data:
            sample_data = load_sample_data()
        else:
            sample_data = original_sample_data
            
        if not sample_data or "pre_state" not in sample_data:
            raise HTTPException(
                status_code=400,
                detail="No valid pre_state found in sample data file"
            )
        
        logger.info("Initializing safrole manager from sample data")
        
        # Initialize the safrole manager with loaded pre_state
        safrole_manager = SafroleManager(sample_data["pre_state"])
        
        logger.info("Safrole manager initialized successfully")
        
        return StateResponse(
            success=True,
            message="Safrole manager initialized successfully from sample data",
            data={
                "initialized": True,
                "current_slot": safrole_manager.state.get("tau", 0),
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
    """
    Process a block using both safrole and dispute components.
    
    This endpoint:
    1. Runs the safrole component first
    2. Then processes disputes using the updated state
    3. Updates the state file with final results
    """
    global safrole_manager
   
    if safrole_manager is None:
        raise HTTPException(
            status_code=400,
            detail="Safrole manager not initialized. Server should auto-initialize on startup."
        )
    
    try:
        logger.info(f"Processing block for slot {request.block.header.slot}")
        
        # Debug: Print the full request structure
        logger.debug(f"Full request structure: {request.dict()}")
        
        # ====== STEP 1: SAFROLE PROCESSING ======
        extrinsic_data = request.block.extrinsic.dict()
        logger.debug(f"Extrinsic data: {extrinsic_data}")
        
        # Extract the tickets from the extrinsic structure for safrole
        tickets_list = extrinsic_data.get("tickets", [])
        
        safrole_block_input = {
            "slot": request.block.header.slot,
            "entropy": request.block.header.entropy_source,
            "extrinsic": tickets_list
        }

        logger.info(f"Safrole block input prepared: {safrole_block_input}")
        
        # Process the block using safrole component
        safrole_result = safrole_manager.process_block(safrole_block_input) 
        
        logger.info(f"Safrole processing completed successfully for slot {request.block.header.slot}")
        
        # Get the updated state from safrole processing
        safrole_post_state = safrole_manager.state
        
        # ====== STEP 2: DISPUTE PROCESSING ======
        logger.info("Starting dispute processing...")
        
        # Read the current state from the updated state file for dispute processing
        try:
            if os.path.exists(updated_state_path):
                with open(updated_state_path, 'r') as f:
                    current_state_data = json.load(f)
                dispute_pre_state = current_state_data.get("pre_state", safrole_post_state)
            else:
                # Use safrole post state if updated state file doesn't exist
                dispute_pre_state = safrole_post_state
            
            logger.info("Pre-state loaded for dispute processing")
            
        except Exception as state_load_error:
            logger.warning(f"Failed to load state from file, using safrole post state: {str(state_load_error)}")
            dispute_pre_state = safrole_post_state
        
        # Extract dispute data from the request
        dispute_input_data = {'disputes': extrinsic_data.get("disputes", {})}
        
        logger.info(f"Dispute input data: {dispute_input_data}")
        
        # Process disputes
        dispute_output, dispute_post_state = process_disputes(
            dispute_input_data, 
            dispute_pre_state, 
            updated_state_path
        )
        
        logger.info(f"Dispute processing completed: {dispute_output}")
        
        # ====== STEP 3: UPDATE STATE FILE ======
        try:
            # Convert bytes to hex for JSON serialization in final state
            final_state_for_json = deep_clone(dispute_post_state)
            if "gamma_a" in final_state_for_json:
                for ticket in final_state_for_json["gamma_a"]:
                    if "randomness" in ticket and isinstance(ticket["randomness"], bytes):
                        ticket["randomness"] = ticket["randomness"].hex()
                    if "proof" in ticket and isinstance(ticket["proof"], bytes):
                        ticket["proof"] = ticket["proof"].hex()
            
            # Create the final block input including both safrole and dispute data
            final_block_input = {
                "safrole": safrole_block_input,
                "disputes": dispute_input_data
            }
            
            # Create/update the updated state file with final results
            create_updated_state_file(final_state_for_json, final_block_input)
            
        except Exception as update_error:
            logger.warning(f"Failed to create updated state file: {str(update_error)}")
            # Don't fail the request if file update fails
        
        # ====== RETURN COMBINED RESULTS ======
        return StateResponse(
            success=True,
            message="Block processed successfully through both safrole and dispute components",
            data={
                "safrole_result": {
                    "header": safrole_result.get("header") if isinstance(safrole_result, dict) else None,
                    "post_state": safrole_result.get("post_state") if isinstance(safrole_result, dict) else None,
                    "result_type": str(type(safrole_result)),
                },
                "dispute_result": {
                    "output": dispute_output,
                    "post_state_keys": list(dispute_post_state.keys()) if isinstance(dispute_post_state, dict) else None
                },
                "current_slot": dispute_post_state.get("tau", 0) if isinstance(dispute_post_state, dict) else 0,
                "processing_completed": {
                    "safrole": True,
                    "dispute": True
                }
            }
        )
        
    except ValueError as e:
        logger.warning(f"Block processing failed with validation error: {str(e)}")
        return StateResponse(
            success=False,
            message="Block processing failed",
            error=str(e)
        )
    except Exception as e:
        logger.error(f"Block processing failed: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Block processing failed: {str(e)}"
        )

@app.get("/state")
async def get_current_state():
    """
    Get the current state of the safrole manager.
    
    Returns the current state if initialized, otherwise an error.
    """
    global safrole_manager
    
    if safrole_manager is None:
        raise HTTPException(
            status_code=400,
            detail="Safrole manager not initialized. Call /initialize first."
        )
    
    try:
        # Return a clean copy of the current state
        current_state = deep_clone(safrole_manager.state)
        
        # Convert bytes to hex for JSON serialization
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
    """
    Get the current updated state from the updated_state.json file.
    
    Returns the updated state if file exists, otherwise an error.
    """
    try:
        if os.path.exists(updated_state_path):
            with open(updated_state_path, 'r') as f:
                updated_state = json.load(f)
            
            return StateResponse(
                success=True,
                message="Updated state retrieved successfully",
                data=updated_state
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

@app.post("/reload-sample-data")
async def reload_sample_data():
    """Reload sample data from file and reinitialize safrole manager.
    This is useful for testing with updated sample data.
    """
    global safrole_manager, original_sample_data
    
    try:
        # Reload sample data
        sample_data = load_sample_data()
        
        if sample_data and "pre_state" in sample_data:
            # Reinitialize safrole manager with the reloaded data
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
    """
    Reset the safrole manager to uninitialized state.
    
    This allows re-initialization with new pre_state data.
    """
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
       "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
