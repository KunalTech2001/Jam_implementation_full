"""
JAM Safrole Integration Server

This server provides REST API endpoints to interact with the JAM protocol
safrole component, allowing state management and block processing.
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
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
sample_data_path = os.path.join(script_dir, "sample_data.json")
updated_state_path = os.path.join(script_dir, "updated_state.json")
original_sample_data: Dict[str, Any] = {}

# Pydantic models for request/response validation
class BlockInput(BaseModel):
    slot: int = Field(..., description="Block slot number")
    entropy: str = Field(..., description="VRF output entropy")
    extrinsic: List[Dict[str, Any]] = Field(default=[], description="Extrinsic data")

# Remove the PreState and StateRequest models since we're loading from file
# Keep only the models we actually need

class BlockProcessRequest(BaseModel):
    input: BlockInput

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
    logger.info("Starting JAM Safrole Integration Server...")
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
    Process a block using the safrole component.
    
    This endpoint takes only the input data and uses the current state
    from the safrole manager to process the block and return results.
    """
    global safrole_manager
   
    if safrole_manager is None:
        raise HTTPException(
            status_code=400,
            detail="Safrole manager not initialized. Server should auto-initialize on startup."
        )
    
    try:
        logger.info(f"Processing block for slot {request.input.slot}")
        
        # Convert request to the format expected by process_block
        block_input = {
            "slot": request.input.slot,
            "entropy": request.input.entropy,
            "extrinsic": request.input.extrinsic
        }

        
        # Process the block using current state
        result = safrole_manager.process_block(block_input) 
        
        logger.info(f"Block processed successfully for slot {request.input.slot}")
        
        # Create/update the updated_state.json file with new state
        try:
            # Get the current state after processing
            current_state = safrole_manager.state
            
            # Convert bytes to hex for JSON serialization
            state_for_json = deep_clone(current_state)
            if "gamma_a" in state_for_json:
                for ticket in state_for_json["gamma_a"]:
                    if "randomness" in ticket and isinstance(ticket["randomness"], bytes):
                        ticket["randomness"] = ticket["randomness"].hex()
                    if "proof" in ticket and isinstance(ticket["proof"], bytes):
                        ticket["proof"] = ticket["proof"].hex()
            
            # Create the updated state file
            create_updated_state_file(state_for_json, block_input)
            
        except Exception as update_error:
            logger.warning(f"Failed to create updated state file: {str(update_error)}")
            # Don't fail the request if file update fails
        
        return StateResponse(
            success=True,
            message="Block processed successfully",
            data={
                "header": result["header"],
                "post_state": result["post_state"],
                "current_slot": safrole_manager.state.get("tau", 0)
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