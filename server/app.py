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

# Global safrole manager instance
safrole_manager: Optional[SafroleManager] = None

# Pydantic models for request/response validation
class BlockInput(BaseModel):
    slot: int = Field(..., description="Block slot number")
    entropy: str = Field(..., description="VRF output entropy")
    extrinsic: List[Dict[str, Any]] = Field(default=[], description="Extrinsic data")

class PreState(BaseModel):
    tau: int = Field(..., description="Current slot number")
    eta: List[str] = Field(..., description="Eta values")
    lambda_: List[Dict[str, str]] = Field(..., description="Lambda values")
    kappa: List[Dict[str, str]] = Field(..., description="Kappa values")
    gamma_k: List[Dict[str, str]] = Field(..., description="Gamma K values")
    gamma_z: str = Field(..., description="Gamma Z value")
    iota: List[Dict[str, str]] = Field(..., description="Iota values")
    gamma_a: List[Dict[str, Any]] = Field(default=[], description="Gamma A values")
    gamma_s: Dict[str, Any] = Field(..., description="Gamma S values")
    post_offenders: List[Any] = Field(default=[], description="Post offenders")

class StateRequest(BaseModel):
    input: BlockInput
    pre_state: PreState

class StateResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the server on startup."""
    global safrole_manager
    logger.info("Starting JAM Safrole Integration Server...")
    logger.info("Server initialized successfully")

@app.get("/")
async def root():
    """Root endpoint with server information."""
    return {
        "message": "JAM Safrole Integration Server",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "safrole_initialized": safrole_manager is not None}

@app.post("/initialize", response_model=StateResponse)
async def initialize_safrole(request: StateRequest):
    """
    Initialize the safrole manager with pre_state data.
    
    This endpoint sets up the safrole component with the provided
    pre_state configuration, making it ready for block processing.
    """
    global safrole_manager
    
    try:
        logger.info(f"Initializing safrole manager with slot {request.input.slot}")
        
        # Convert pre_state to the format expected by SafroleManager
        pre_state_dict = request.pre_state.dict()
        
        # Initialize the safrole manager
        safrole_manager = SafroleManager(pre_state_dict)
        
        logger.info("Safrole manager initialized successfully")
        
        return StateResponse(
            success=True,
            message="Safrole manager initialized successfully",
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
async def process_block(request: StateRequest):
    """
    Process a block using the safrole component.
    
    This endpoint takes the input data and processes it through
    the safrole manager to update the state and return results.
    """
    global safrole_manager
   
    if safrole_manager is None:
        raise HTTPException(
            status_code=400,
            detail="Safrole manager not initialized. Call /initialize first."
        )
    
    try:
        logger.info(f"Processing block for slot {request.input.slot}")
        
        # Convert request to the format expected by process_block
        block_input = {
            "slot": request.header.slot,
            "entropy": request.header.entropy,
            "extrinsic": request.block.extrinsic
        }

        print(block_input)
        
        # Process the block
        result = safrole_manager.process_block(block_input) 
        
        logger.info(f"Block processed successfully for slot {request.input.slot}")
        
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
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
