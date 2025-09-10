"""
Implements the Refine process, taking a Work-Package and Refinement Context
to produce a Work-Report. This embodies the Ξ function.
"""

import json
import logging
import random
from typing import List, Optional

from models import WorkPackage, RefinementContext, WorkReport, AvailabilitySpec
from utils.errors import PVMExecutionError, AuthorizationError
from offchain.signature import sign_message, public_key_to_base64
from offchain.encoder import encode_for_availability
from models.work_digest import WorkDigest

# Set up logging
logging.basicConfig(level=logging.INFO)

class PVMResult:
    def __init__(self, output: str, gas_used: int):
        self.output = output
        self.gas_used = gas_used

class OnchainState:
    def __init__(self):
        self.ξ = set()
        self.ρ = set()

def simulate_vr_pvm(work_package: WorkPackage, context: RefinementContext, force_gas_used: Optional[int] = None) -> PVMResult:
    try:
        combined_input = json.dumps({
            "workPackageId": work_package.authorization_token,
            "contextAnchor": context.anchor_block_root,
            "firstWorkItemProgram": work_package.work_items[0].program_hash if work_package.work_items else None,
            "firstWorkItemInput": work_package.work_items[0].input_data if work_package.work_items else None,
        }, sort_keys=True)

        # Simulate output and gas used
        simulated_output = f"PVM_OUTPUT_{WorkDigest.sha256_hash(combined_input)}"
        simulated_gas_used = force_gas_used if force_gas_used is not None else random.randint(100, 1099)

        # Simulate error with 0% probability (as in TS)
        # if random.random() < 0.0:
        #     raise PVMExecutionError('Simulated PVM execution failed.')

        return PVMResult(simulated_output, simulated_gas_used)
    except Exception as error:
        raise PVMExecutionError(f"PVM simulation error: {str(error)}")

def historical_lookup(dependency_hashes: List[str], onchain_state: OnchainState) -> bool:
    logging.info(f"Performing historical lookup for dependencies: {', '.join(dependency_hashes)}")
    for dep_hash in dependency_hashes:
        if dep_hash not in onchain_state.ξ and dep_hash not in onchain_state.ρ:
            logging.warning(f"Dependency {dep_hash} not found in history or pending reports.")
            return False
    return True

def check_authorization(authorization_token: str, auth_service_details: dict) -> bool:
    logging.info(f"Checking authorization for token: {authorization_token} via service: {auth_service_details.get('h')}")
    return True

def refine_work_package(
    work_package: WorkPackage,
    refinement_context: RefinementContext,
    guarantor_private_key: bytes,
    core_index: int,
    slot: int,
    dependencies: List[str],
    onchain_state: OnchainState,
    force_gas_used: Optional[int] = None
) -> WorkReport:
    logging.info("Starting refinement process...")

    # 1. Authorization Check
    if not check_authorization(work_package.authorization_token, work_package.authorization_service_details):
        raise AuthorizationError("Work-Package not authorized.")

    # 2. Historical Lookup for Dependencies
    if dependencies and not historical_lookup(dependencies, onchain_state):
        raise Exception("One or more dependencies not found in historical state.")

    # 3. Simulate V_R PVM Execution
    try:
        pvm_result = simulate_vr_pvm(work_package, refinement_context, force_gas_used)
        logging.info(f"PVM executed. Output: {pvm_result.output}, Gas Used: {pvm_result.gas_used}")
    except Exception as error:
        raise PVMExecutionError(f"Failed to execute PVM: {str(error)}")

    # 4. Generate Availability Specification
    availability_spec = encode_for_availability(
        WorkReport(
            work_package,
            refinement_context,
            pvm_result.output,
            pvm_result.gas_used,
            None,
            '',
            '',
            core_index,
            slot,
            dependencies
        ),
        4,
        2
    )
    logging.info("Availability Specification generated.")

    # 5. Construct Work-Report (Preliminary)
    # For Ed25519, public key is the last 32 bytes of the 64-byte private key
    guarantor_public_key = public_key_to_base64(guarantor_private_key[-32:])
    preliminary_report = WorkReport(
        work_package,
        refinement_context,
        pvm_result.output,
        pvm_result.gas_used,
        availability_spec,
        '',
        guarantor_public_key,
        core_index,
        slot,
        dependencies
    )

    # 6. Sign the Work-Report
    signature = sign_message(preliminary_report.to_signable_object(), guarantor_private_key)
    logging.info("Work-Report signed.")

    # 7. Finalize Work-Report with Signature
    preliminary_report.guarantor_signature = signature

    logging.info("Refinement process completed. Work-Report generated.")
    return preliminary_report