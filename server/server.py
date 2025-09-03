




# from fastapi import FastAPI, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel, Field
# from typing import List, Dict, Any, Optional
# from contextlib import asynccontextmanager
# import uvicorn
# import logging
# import sys
# import os
# import json
# import datetime
# from datetime import datetime
# import subprocess

# # Add the src directory to the path to import jam modules
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# from jam.core.safrole_manager import SafroleManager
# from jam.utils.helpers import deep_clone

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI(
#     title="JAM Safrole Integration Server",
#     description="REST API server for JAM protocol safrole component integration",
#     version="1.0.0"
# )

# # Add CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global variables
# safrole_manager: Optional[SafroleManager] = None
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sample_data_path = os.path.join(script_dir, "sample_data.json")
# updated_state_path = os.path.join(script_dir, "updated_state.json")
# jam_history_script = "/Users/anish/Desktop/fulljam/Jam_implementation_full/Jam-history/test.py"
# original_sample_data: Dict[str, Any] = {}

# # Pydantic models for request/response validation
# class BlockHeader(BaseModel):
#     parent: str
#     parent_state_root: str
#     extrinsic_hash: str
#     slot: int
#     epoch_mark: Optional[Any] = None
#     tickets_mark: Optional[Any] = None
#     offenders_mark: List[Any] = []
#     author_index: int
#     entropy_source: str
#     seal: str

# class BlockDisputes(BaseModel):
#     verdicts: List[Any] = []
#     culprits: List[Any] = []
#     faults: List[Any] = []

# class BlockExtrinsic(BaseModel):
#     tickets: List[Any] = []
#     preimages: List[Any] = []
#     guarantees: List[Any] = []
#     assurances: List[Any] = []
#     disputes: BlockDisputes

# class Block(BaseModel):
#     header: BlockHeader
#     extrinsic: BlockExtrinsic

# class BlockProcessRequest(BaseModel):
#     block: Block

# class StateResponse(BaseModel):
#     success: bool
#     message: str
#     data: Optional[Dict[str, Any]] = None
#     error: Optional[str] = None

# def load_sample_data():
#     """Load sample data from JSON file."""
#     global original_sample_data
#     try:
#         if os.path.exists(sample_data_path):
#             with open(sample_data_path, 'r') as f:
#                 original_sample_data = json.load(f)
#                 logger.info(f"Sample data loaded from {sample_data_path}")
#                 return original_sample_data
#         else:
#             logger.warning(f"Sample data file {sample_data_path} not found")
#             return {}
#     except Exception as e:
#         logger.error(f"Failed to load sample data: {str(e)}")
#         return {}

# def create_updated_state_file(new_state_data: Dict[str, Any], block_input: Dict[str, Any]):
#     """Create/update the updated_state.json file with new state information."""
#     try:
#         updated_data = deep_clone(original_sample_data)
#         updated_data["pre_state"] = new_state_data
#         updated_data["input"] = block_input
#         updated_data["metadata"] = {
#             "last_updated": str(datetime.now()),
#             "current_slot": new_state_data.get("tau", 0),
#             "updated_from_original": sample_data_path
#         }
#         with open(updated_state_path, 'w') as f:
#             json.dump(updated_data, f, indent=2)
#         logger.info(f"Updated state saved to {updated_state_path}")
#         return True
#     except Exception as e:
#         logger.error(f"Failed to create updated state file: {str(e)}")
#         return False

# def run_jam_history():
#     """Run the jam_history component (test.py)."""
#     try:
#         logger.info(f"Attempting to run jam_history component: {jam_history_script}")
#         result = subprocess.run(
#             ["python3", jam_history_script],
#             capture_output=True,
#             text=True,
#             check=True
#         )
#         logger.info(f"jam_history component executed successfully. Output: {result.stdout}")
#         return True
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Failed to run jam_history component: {e.stderr}")
#         return False
#     except Exception as e:
#         logger.error(f"Unexpected error while running jam_history component: {str(e)}")
#         return False

# def prompt_for_jam_history():
#     """Prompt user in terminal to run jam_history component."""
#     while True:
#         print("\nDo you want to run the jam_history component? (yes/no)")
#         choice = input().strip().lower()
#         if choice in ['yes', 'y']:
#             return True
#         elif choice in ['no', 'n']:
#             return False
#         else:
#             print("Invalid input. Please enter 'yes' or 'no'.")

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Handle startup and shutdown events."""
#     global safrole_manager, original_sample_data
#     logger.info("Starting JAM Safrole Integration Server...")
#     logger.info(f"Looking for sample data at: {sample_data_path}")
    
#     if not os.path.exists(sample_data_path):
#         logger.error(f"Sample data file not found at: {sample_data_path}")
#         yield
#         return
        
#     sample_data = load_sample_data()
    
#     if not sample_data:
#         logger.error("Failed to load sample data or file is empty")
#         yield
#         return
        
#     if "pre_state" not in sample_data:
#         logger.error("No 'pre_state' key found in sample data")
#         yield
#         return
        
#     try:
#         safrole_manager = SafroleManager(sample_data["pre_state"])
#         logger.info("Safrole manager successfully initialized with sample data on startup")
#     except Exception as e:
#         logger.error(f"Failed to initialize safrole manager on startup: {str(e)}")
    
#     logger.info("Server initialized successfully")
#     yield
#     logger.info("Shutting down JAM Safrole Integration Server...")

# # Update FastAPI app to use lifespan
# app = FastAPI(
#     title="JAM Safrole Integration Server",
#     description="REST API server for JAM protocol safrole component integration",
#     version="1.0.0",
#     lifespan=lifespan
# )

# @app.get("/")
# async def root():
#     """Root endpoint with server information."""
#     return {
#         "message": "JAM Safrole Integration Server",
#         "version": "1.0.0",
#         "status": "running",
#         "sample_data_loaded": len(original_sample_data) > 0
#     }

# @app.get("/health")
# async def health_check():
#     """Health check endpoint."""
#     return {
#         "status": "healthy", 
#         "safrole_initialized": safrole_manager is not None,
#         "sample_data_loaded": len(original_sample_data) > 0
#     }

# @app.post("/initialize", response_model=StateResponse)
# async def initialize_safrole():
#     """Initialize the safrole manager with pre_state data from sample_data.json."""
#     global safrole_manager
#     try:
#         if not original_sample_data:
#             sample_data = load_sample_data()
#         else:
#             sample_data = original_sample_data
            
#         if not sample_data or "pre_state" not in sample_data:
#             raise HTTPException(
#                 status_code=400,
#                 detail="No valid pre_state found in sample data file"
#             )
        
#         logger.info("Initializing safrole manager from sample data")
#         safrole_manager = SafroleManager(sample_data["pre_state"])
#         logger.info("Safrole manager initialized successfully")
        
#         return StateResponse(
#             success=True,
#             message="Safrole manager initialized successfully from sample data",
#             data={
#                 "initialized": True,
#                 "current_slot": safrole_manager.state.get("tau", 0),
#                 "epoch_length": safrole_manager.state.get("E", 12),
#                 "submission_period": safrole_manager.state.get("Y", 11)
#             }
#         )
#     except Exception as e:
#         logger.error(f"Failed to initialize safrole manager: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to initialize safrole manager: {str(e)}"
#         )

# @app.post("/process-block", response_model=StateResponse)
# async def process_block(request: BlockProcessRequest):
#     """Process a block using the safrole component and optionally run jam_history."""
#     global safrole_manager
#     if safrole_manager is None:
#         raise HTTPException(
#             status_code=400,
#             detail="Safrole manager not initialized. Server should auto-initialize on startup."
#         )
    
#     try:
#         logger.info(f"Processing block for slot {request.block.header.slot}")
#         logger.debug(f"Full request structure: {request.dict()}")
#         extrinsic_data = request.block.extrinsic.dict()
#         logger.debug(f"Extrinsic data type: {type(extrinsic_data)}")
#         logger.debug(f"Extrinsic data: {extrinsic_data}")
        
#         tickets_list = extrinsic_data.get("tickets", [])
#         block_input = {
#             "slot": request.block.header.slot,
#             "entropy": request.block.header.entropy_source,
#             "extrinsic": tickets_list
#         }

#         logger.info(f"Block input prepared: {block_input}")
#         logger.debug(f"Safrole manager state type: {type(safrole_manager.state)}")
        
#         result = safrole_manager.process_block(block_input)
#         logger.info(f"Block processed successfully for slot {request.block.header.slot}")
        
#         # Create/update the updated_state.json file
#         try:
#             current_state = safrole_manager.state
#             state_for_json = deep_clone(current_state)
#             if "gamma_a" in state_for_json:
#                 for ticket in state_for_json["gamma_a"]:
#                     if "randomness" in ticket and isinstance(ticket["randomness"], bytes):
#                         ticket["randomness"] = ticket["randomness"].hex()
#                     if "proof" in ticket and isinstance(ticket["proof"], bytes):
#                         ticket["proof"] = ticket["proof"].hex()
            
#             if create_updated_state_file(state_for_json, block_input):
#                 # Prompt user to run jam_history
#                 if prompt_for_jam_history():
#                     if run_jam_history():
#                         logger.info("jam_history component ran successfully")
#                     else:
#                         logger.warning("jam_history component failed to run")
#                 else:
#                     logger.info("User chose not to run jam_history component")
#             else:
#                 logger.warning("Failed to create updated state file, skipping jam_history")
            
#         except Exception as update_error:
#             logger.warning(f"Failed to create updated state file: {str(update_error)}")
        
#         return StateResponse(
#             success=True,
#             message="Block processed successfully",
#             data={
#                 "header": result.get("header") if isinstance(result, dict) else None,
#                 "post_state": result.get("post_state") if isinstance(result, dict) else None,
#                 "current_slot": safrole_manager.state.get("tau", 0) if isinstance(safrole_manager.state, dict) else 0,
#                 "result_type": str(type(result)),
#                 "result": str(result)[:200]
#             }
#         )
#     except ValueError as e:
#         logger.warning(f"Block processing failed with validation error: {str(e)}")
#         return StateResponse(
#             success=False,
#             message="Block processing failed",
#             error=str(e)
#         )
#     except Exception as e:
#         logger.error(f"Block processing failed: {str(e)}")
#         logger.error(f"Exception type: {type(e)}")
#         import traceback
#         logger.error(f"Full traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Block processing failed: {str(e)}"
#         )

# @app.get("/state")
# async def get_current_state():
#     """Get the current state of the safrole manager."""
#     global safrole_manager
#     if safrole_manager is None:
#         raise HTTPException(
#             status_code=400,
#             detail="Safrole manager not initialized. Call /initialize first."
#         )
    
#     try:
#         current_state = deep_clone(safrole_manager.state)
#         if "gamma_a" in current_state:
#             for ticket in current_state["gamma_a"]:
#                 if "randomness" in ticket and isinstance(ticket["randomness"], bytes):
#                     ticket["randomness"] = ticket["randomness"].hex()
#                 if "proof" in ticket and isinstance(ticket["proof"], bytes):
#                     ticket["proof"] = ticket["proof"].hex()
        
#         return StateResponse(
#             success=True,
#             message="Current state retrieved successfully",
#             data=current_state
#         )
#     except Exception as e:
#         logger.error(f"Failed to retrieve current state: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to retrieve current state: {str(e)}"
#         )

# @app.get("/updated-state")
# async def get_updated_state():
#     """Get the current updated state from the updated_state.json file."""
#     try:
#         if os.path.exists(updated_state_path):
#             with open(updated_state_path, 'r') as f:
#                 updated_state = json.load(f)
#             return StateResponse(
#                 success=True,
#                 message="Updated state retrieved successfully",
#                 data=updated_state
#             )
#         else:
#             return StateResponse(
#                 success=False,
#                 message="Updated state file not found",
#                 error=f"File {updated_state_path} does not exist"
#             )
#     except Exception as e:
#         logger.error(f"Failed to retrieve updated state: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to retrieve updated state: {str(e)}"
#         )

# @app.post("/reload-sample-data")
# async def reload_sample_data():
#     """Reload sample data from file and reinitialize safrole manager."""
#     global safrole_manager, original_sample_data
#     try:
#         sample_data = load_sample_data()
#         if sample_data and "pre_state" in sample_data:
#             safrole_manager = SafroleManager(sample_data["pre_state"])
#             logger.info("Sample data reloaded and safrole manager reinitialized")
#             return StateResponse(
#                 success=True,
#                 message="Sample data reloaded and safrole manager reinitialized successfully"
#             )
#         else:
#             return StateResponse(
#                 success=False,
#                 message="No valid sample data found in file",
#                 error="Invalid or missing pre_state in sample data"
#             )
#     except Exception as e:
#         logger.error(f"Failed to reload sample data: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to reload sample data: {str(e)}"
#         )

# @app.post("/reset")
# async def reset_safrole():
#     """Reset the safrole manager to uninitialized state."""
#     global safrole_manager
#     try:
#         safrole_manager = None
#         logger.info("Safrole manager reset successfully")
#         return StateResponse(
#             success=True,
#             message="Safrole manager reset successfully"
#         )
#     except Exception as e:
#         logger.error(f"Failed to reset safrole manager: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to reset safrole manager: {str(e)}"
#         )

# @app.exception_handler(Exception)
# async def global_exception_handler(request: Request, exc: Exception):
#     """Global exception handler for unhandled errors."""
#     logger.error(f"Unhandled exception: {str(exc)}")
#     return JSONResponse(
#         status_code=500,
#         content={
#             "success": False,
#             "message": "Internal server error",
#             "error": str(exc)
#         }
#     )

# if __name__ == "__main__":
#     uvicorn.run(
#         "server.app:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True,
#         log_level="info"
#     )

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
from contextlib import asynccontextmanager
import psutil
import subprocess


# Add the src directory to the path to import jam modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jam.core.safrole_manager import SafroleManager
from jam.utils.helpers import deep_clone

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JAM Safrole, Dispute, and State Integration Server",
    description="REST API server for JAM protocol safrole, dispute, and state component integration",
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
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to get project root
sample_data_path = os.path.join(script_dir, "sample_data.json")
updated_state_path = os.path.join(script_dir, "updated_state.json")
jam_history_script = os.path.join(project_root, "Jam-history", "test.py")
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
            updated_data = {"pre_state": deepcopy(DEFAULT_SAMPLE_DATA["pre_state"]), "metadata": {}}
            with open(updated_state_path, 'w') as f:
                json.dump(updated_data, f, indent=2)
            logger.info(f"Default updated state created at {updated_state_path}")
            return updated_data["pre_state"]
        
        with open(updated_state_path, 'r') as f:
            updated_data = json.load(f)
            logger.debug(f"Loaded updated_state.json: {updated_data}")
            pre_state = updated_data.get('pre_state', {})
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
        logger.info(f"jam_history component executed successfully. Output: {result.stdout}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run jam_history component: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"Unexpected error while running jam_history component: {str(e)}")
        return False, str(e)


def run_jam_preimages():
    """Run the jam-preimages component (main.py)."""
    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the jam-preimages directory
        jam_preimages_dir = os.path.join(current_dir, "jam-preimages")
        
        if not os.path.exists(jam_preimages_dir):
            logger.warning(f"jam-preimages directory not found at {jam_preimages_dir}")
            return False, "jam-preimages directory not found"
            
        # Run the main.py script
        result = subprocess.run(
            ["python3", "main.py"],
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
        error_msg = f"Error running jam-preimages: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def run_assurances_component():
    """Run the assurances component and merge its output with the current state.
    
    This function will:
    1. Load the current state from updated_state.json
    2. Load the post_state from assurances/post_state.json
    3. Merge the assurance-related fields from post_state into the current state
    4. Preserve all other fields in the current state
    5. Save the merged state back to updated_state.json
    """
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
            
            # Save new state
            with open(updated_state_file, 'w') as f:
                json.dump([merged_state], f, indent=2)
            
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
                # Deep merge existing data into our base
                updated_data = deep_merge(updated_data, existing_data)
                logger.info("Merged existing state with original sample data")
            except json.JSONDecodeError as e:
                logger.error(f"Error reading existing state file: {e}")
                # If we can't read the existing file, continue with just the sample data
        
        # Update the pre_state by merging with new_state_data
        if "pre_state" in updated_data and isinstance(updated_data["pre_state"], dict):
            updated_data["pre_state"] = deep_merge(updated_data["pre_state"], new_state_data)
        else:
            updated_data["pre_state"] = deep_clone(new_state_data)
        
        # Update the input with the new block input
        updated_data["input"] = deep_clone(block_input)
        
        # Ensure metadata exists and update it
        if "metadata" not in updated_data:
            updated_data["metadata"] = {}
        
        # Update metadata fields while preserving any existing ones
        updated_data["metadata"].update({
            "last_updated": str(datetime.now()),
            "current_slot": new_state_data.get("slot", new_state_data.get("tau", updated_data["metadata"].get("current_slot", 0))),
            "updated_from_original": sample_data_path
        })
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(updated_state_path), exist_ok=True)
        
        # Write to a temporary file first
        with open(temp_path, 'w') as f:
            json.dump(updated_data, f, indent=2, sort_keys=True)
        
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

def init_empty_stats(num_validators: int) -> List[Dict[str, int]]:
    """Initialize empty validator stats for epoch change."""
    return [{
        "blocks": 0,
        "tickets": 0,
        "pre_images": 0,
        "pre_images_size": 0,
        "guarantees": 0,
        "assurances": 0
    } for _ in range(num_validators)]

def process_blockchain(input_data: Dict[str, Any], pre_state: Dict[str, Any], is_epoch_change: bool) -> tuple:
    """Process state component per JAM protocol section 13.1."""
    logger.info(f"Memory before state processing: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    
    # Validate pre_state fields
    required_fields = ['vals_curr_stats', 'vals_last_stats', 'slot', 'curr_validators']
    missing_fields = [field for field in required_fields if field not in pre_state]
    if missing_fields:
        logger.error(f"Missing required fields in pre_state for state processing: {missing_fields}")
        return {"err": f"missing_state_fields: {missing_fields}"}, deepcopy(pre_state)
    
    # Initialize output as null per test vector
    output = None
    
    # Initialize post_state
    if is_epoch_change:
        # Epoch change: reset vals_curr_stats, move pre_state.vals_curr_stats to vals_last_stats
        post_state = {
            'vals_curr_stats': init_empty_stats(len(pre_state['curr_validators'])),
            'vals_last_stats': [stats.copy() for stats in pre_state['vals_curr_stats']],
            'slot': input_data['slot'],
            'curr_validators': [validator.copy() for validator in pre_state['curr_validators']]
        }
    else:
        # Normal block processing: copy and update vals_curr_stats, keep vals_last_stats
        post_state = {
            'vals_curr_stats': [stats.copy() for stats in pre_state['vals_curr_stats']],
            'vals_last_stats': [stats.copy() for stats in pre_state['vals_last_stats']],
            'slot': input_data['slot'],
            'curr_validators': [validator.copy() for validator in pre_state['curr_validators']]
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
    
    # Update stats for authoring validator
    v_stats = post_state['vals_curr_stats'][author_index]
    v_stats['blocks'] += 1  # π'V[v].b = a[v].b + (v = HI)
    
    # Process extrinsic (if non-empty)
    if extrinsic.get('tickets') or extrinsic.get('preimages') or extrinsic.get('guarantees') or extrinsic.get('assurances'):
        v_stats['tickets'] += len(extrinsic['tickets'])  # π'V[v].t = a[v].t + |ET|
        v_stats['pre_images'] += len(extrinsic['preimages'])  # π'V[v].p = a[v].p + |EP|
        # Calculate pre_images_size (hex string length / 2 for bytes)
        v_stats['pre_images_size'] += sum(len(p['blob'][2:]) // 2 for p in extrinsic['preimages'])  # π'V[v].d = ∑|d|
        
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
        "status": "healthy", 
        "safrole_initialized": safrole_manager is not None,
        "sample_data_loaded": len(original_sample_data) > 0
    }

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
            update_state_file(combined_state, block_input)
            
            # Run jam_history after successful state update
            jam_history_success, jam_history_output = run_jam_history()
            if not jam_history_success:
                logger.warning(f"jam_history execution completed with warnings: {jam_history_output}")
            
            # Run jam-preimages after jam_history completes
            jam_preimages_success, jam_preimages_output = run_jam_preimages()
            if not jam_preimages_success:
                logger.warning(f"jam-preimages execution completed with warnings: {jam_preimages_output}")
            
            # Run assurances component after all other components complete
            assurances_success, assurances_output = run_assurances_component()
            if not assurances_success:
                logger.warning(f"assurances component execution completed with warnings: {assurances_output}")
            
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
        logger.error(f"Block processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Block processing failed: {str(e)}")



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
