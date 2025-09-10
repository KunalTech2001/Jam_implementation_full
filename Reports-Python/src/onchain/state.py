from typing import Dict, Set, Any, Optional

# Placeholder imports for actual model classes
# from ..models.work_report import WorkReport
# from ..models.work_digest import WorkDigest

class PendingReportEntry:
    def __init__(self, report, received_signatures: Set[str], submission_slot: int):
        self.report = report
        self.received_signatures = set(received_signatures)
        self.submission_slot = submission_slot

class AccumulationEntry:
    def __init__(self, report, status: str):
        self.report = report
        self.status = status  # 'pending', 'ready', or 'processing'

class BadReportEntry:
    def __init__(self, reason: str, disputed_by: Set[str]):
        self.reason = reason
        self.disputed_by = set(disputed_by)

class OffenderEntry:
    def __init__(self, dispute_count: int, last_dispute_slot: int):
        self.dispute_count = dispute_count
        self.last_dispute_slot = last_dispute_slot

class AccountState(dict):
    def __init__(self, balance: float = 0, **kwargs):
        super().__init__(balance=balance, **kwargs)

class GlobalState:
    def __init__(
        self,
        accounts: Optional[Dict[str, 'AccountState']] = None,
        core_status: Optional[Dict[int, str]] = None,
        service_registry: Optional[Dict[str, dict]] = None
    ):
        self.accounts = accounts if accounts is not None else {}
        self.core_status = core_status if core_status is not None else {}
        self.service_registry = service_registry if service_registry is not None else {}

class OnchainState:
    def __init__(self):
        self.ρ: Dict[str, PendingReportEntry] = {}
        self.ω: Dict[str, AccumulationEntry] = {}
        self.ξ: Dict[str, Any] = {}  # WorkReport
        self.ψ_B: Dict[str, BadReportEntry] = {}
        self.ψ_O: Dict[str, OffenderEntry] = {}
        self.global_state: GlobalState = GlobalState()

    def reset(self):
        self.ρ.clear()
        self.ω.clear()
        self.ξ.clear()
        self.ψ_B.clear()
        self.ψ_O.clear()
        self.global_state = GlobalState()

    def get_report_by_digest(self, digest_hash: str):
        if digest_hash in self.ξ:
            return self.ξ[digest_hash]
        if digest_hash in self.ρ:
            return self.ρ[digest_hash].report
        if digest_hash in self.ω:
            return self.ω[digest_hash].report
        return None

    def to_plain_object(self) -> dict:
        def map_to_obj(mapping):
            if isinstance(mapping, dict):
                obj = {}
                for k, v in mapping.items():
                    if isinstance(v, dict):
                        obj[k] = map_to_obj(v)
                    elif isinstance(v, set):
                        obj[k] = list(v)
                    elif hasattr(v, "to_object") and callable(getattr(v, "to_object")):
                        obj[k] = v.to_object()
                    elif hasattr(v, "to_plain_object") and callable(getattr(v, "to_plain_object")):
                        obj[k] = v.to_plain_object()
                    elif hasattr(v, "__dict__"):
                        obj[k] = map_to_obj(vars(v))
                    else:
                        obj[k] = v
                return obj
            elif isinstance(mapping, set):
                return list(mapping)
            else:
                return mapping

        global_state = self.global_state or GlobalState()
        return {
            'ρ': map_to_obj(self.ρ),
            'ω': map_to_obj(self.ω),
            'ξ': map_to_obj(self.ξ),
            'ψ_B': map_to_obj(self.ψ_B),
            'ψ_O': map_to_obj(self.ψ_O),
            'globalState': {
                'accounts': dict(global_state.accounts or {}),
                'coreStatus': dict(global_state.core_status or {}),
                'serviceRegistry': dict(global_state.service_registry or {}),
            }
        }