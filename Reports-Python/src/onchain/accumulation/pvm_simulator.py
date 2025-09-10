"""
Mocks the Ψ_A PVM invocation for state mutation.
This PVM takes a Work-Item and applies its logic to a conceptual global state.
"""

import json
from typing import Dict, Any

# from ...models.work_item import WorkItem
# from ...utils.errors import PVMExecutionError

class PVMExecutionError(Exception):
    pass

def simulate_psi_a_pvm(work_item, current_global_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulates the Ψ_A PVM execution for a single Work-Item.
    :param work_item: The Work-Item to execute.
    :param current_global_state: The current conceptual global state.
    :return: A state delta representing changes to the global state.
    """
    print(f"[Ψ_A PVM] Executing Work-Item: {work_item.id} (Program: {work_item.program_hash})")

    state_delta = {}
    gas_consumed = 0

    try:
        if work_item.program_hash == "0xtransfer":
            from_to = json.loads(work_item.input_data)
            from_acc = from_to.get("from")
            to_acc = from_to.get("to")
            amount = from_to.get("amount")

            accounts = current_global_state.get("accounts", {})
            if (
                from_acc in accounts and
                accounts[from_acc]["balance"] >= amount
            ):
                  # Copy accounts to avoid mutating the original
                new_accounts = dict(accounts)
                new_accounts[from_acc] = dict(new_accounts[from_acc])
                new_accounts[to_acc] = dict(new_accounts.get(to_acc, {"balance": 0}))

                new_accounts[from_acc]["balance"] -= amount
                new_accounts[to_acc]["balance"] += amount

                state_delta["accounts"] = new_accounts
                gas_consumed = 50
            else:
                raise Exception("Insufficient balance or invalid accounts for transfer.")

        elif work_item.program_hash == "0xupdateData":
            kv = json.loads(work_item.input_data)
            key = kv.get("key")
            value = kv.get("value")
            data = dict(current_global_state.get("data", {}))
            data[key] = value
            state_delta["data"] = data
            gas_consumed = 20

        else:
            log = current_global_state.get("log", "")
            state_delta["log"] = (
                log + f"Executed {work_item.program_hash} with input {work_item.input_data}."
            )
            gas_consumed = 10

        if gas_consumed > getattr(work_item, "gas_limit", 0):
            raise Exception(
                f"Gas limit exceeded for Work-Item {work_item.id}. Consumed: {gas_consumed}, Limit: {work_item.gas_limit}"
            )

        print(f"[Ψ_A PVM] Work-Item {work_item.id} executed successfully. Gas consumed: {gas_consumed}")
        return state_delta

    except Exception as error:
        print(f"[Ψ_A PVM] Error executing Work-Item {work_item.id}: {error}")
        raise PVMExecutionError(
            f"Ψ_A PVM execution failed for Work-Item {work_item.id}: {error}"
        )