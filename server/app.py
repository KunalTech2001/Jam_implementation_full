from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
import uvicorn
import logging
import sys
import os
import json
import datetime
from datetime import datetime
import subprocess

# Add the src directory to the path to import jam modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from jam.core.safrole_manager import SafroleManager
from jam.utils.helpers import deep_clone

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="JAM Safrole Integration Server",
    description="REST API server for JAM protocol safrole component integration",
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
sample_data_path = os.path.join(script_dir, "sample_data.json")
updated_state_path = os.path.join(script_dir, "updated_state.json")
jam_history_script = "/Users/anish/Desktop/fulljam/Jam_implementation_full/Jam-history/test.py"
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
        updated_data = deep_clone(original_sample_data)
        updated_data["pre_state"] = new_state_data
        updated_data["input"] = block_input
        updated_data["metadata"] = {
            "last_updated": str(datetime.now()),
            "current_slot": new_state_data.get("tau", 0),
            "updated_from_original": sample_data_path
        }
        with open(updated_state_path, 'w') as f:
            json.dump(updated_data, f, indent=2)
        logger.info(f"Updated state saved to {updated_state_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create updated state file: {str(e)}")
        return False

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
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run jam_history component: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while running jam_history component: {str(e)}")
        return False

def prompt_for_jam_history():
    """Prompt user in terminal to run jam_history component."""
    while True:
        print("\nDo you want to run the jam_history component? (yes/no)")
        choice = input().strip().lower()
        if choice in ['yes', 'y']:
            return True
        elif choice in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    global safrole_manager, original_sample_data
    logger.info("Starting JAM Safrole Integration Server...")
    logger.info(f"Looking for sample data at: {sample_data_path}")
    
    if not os.path.exists(sample_data_path):
        logger.error(f"Sample data file not found at: {sample_data_path}")
        yield
        return
        
    sample_data = load_sample_data()
    
    if not sample_data:
        logger.error("Failed to load sample data or file is empty")
        yield
        return
        
    if "pre_state" not in sample_data:
        logger.error("No 'pre_state' key found in sample data")
        yield
        return
        
    try:
        safrole_manager = SafroleManager(sample_data["pre_state"])
        logger.info("Safrole manager successfully initialized with sample data on startup")
    except Exception as e:
        logger.error(f"Failed to initialize safrole manager on startup: {str(e)}")
    
    logger.info("Server initialized successfully")
    yield
    logger.info("Shutting down JAM Safrole Integration Server...")

# Update FastAPI app to use lifespan
app = FastAPI(
    title="JAM Safrole Integration Server",
    description="REST API server for JAM protocol safrole component integration",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "message": "JAM Safrole Integration Server",
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
    """Process a block using the safrole component and optionally run jam_history."""
    global safrole_manager
    if safrole_manager is None:
        raise HTTPException(
            status_code=400,
            detail="Safrole manager not initialized. Server should auto-initialize on startup."
        )
    
    try:
        logger.info(f"Processing block for slot {request.block.header.slot}")
        logger.debug(f"Full request structure: {request.dict()}")
        extrinsic_data = request.block.extrinsic.dict()
        logger.debug(f"Extrinsic data type: {type(extrinsic_data)}")
        logger.debug(f"Extrinsic data: {extrinsic_data}")
        
        tickets_list = extrinsic_data.get("tickets", [])
        block_input = {
            "slot": request.block.header.slot,
            "entropy": request.block.header.entropy_source,
            "extrinsic": tickets_list
        }

        logger.info(f"Block input prepared: {block_input}")
        logger.debug(f"Safrole manager state type: {type(safrole_manager.state)}")
        
        result = safrole_manager.process_block(block_input)
        logger.info(f"Block processed successfully for slot {request.block.header.slot}")
        
        # Create/update the updated_state.json file
        try:
            current_state = safrole_manager.state
            state_for_json = deep_clone(current_state)
            if "gamma_a" in state_for_json:
                for ticket in state_for_json["gamma_a"]:
                    if "randomness" in ticket and isinstance(ticket["randomness"], bytes):
                        ticket["randomness"] = ticket["randomness"].hex()
                    if "proof" in ticket and isinstance(ticket["proof"], bytes):
                        ticket["proof"] = ticket["proof"].hex()
            
            if create_updated_state_file(state_for_json, block_input):
                # Prompt user to run jam_history
                if prompt_for_jam_history():
                    if run_jam_history():
                        logger.info("jam_history component ran successfully")
                    else:
                        logger.warning("jam_history component failed to run")
                else:
                    logger.info("User chose not to run jam_history component")
            else:
                logger.warning("Failed to create updated state file, skipping jam_history")
            
        except Exception as update_error:
            logger.warning(f"Failed to create updated state file: {str(update_error)}")
        
        return StateResponse(
            success=True,
            message="Block processed successfully",
            data={
                "header": result.get("header") if isinstance(result, dict) else None,
                "post_state": result.get("post_state") if isinstance(result, dict) else None,
                "current_slot": safrole_manager.state.get("tau", 0) if isinstance(safrole_manager.state, dict) else 0,
                "result_type": str(type(result)),
                "result": str(result)[:200]
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
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

