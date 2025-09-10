from utils.validator import validate_required, validate_type

class RefinementContext:
    """
    Represents the Refinement Context (C) provided to the PVM.
    Contains information about the current state of the blockchain.
    """

    def __init__(
        self,
        anchor_block_root: str,
        anchor_block_number: int,
        beefy_mmr_root: str,
        current_slot: int,
        current_epoch: int,
        current_guarantors: list[str],
        previous_guarantors: list[str]
    ):
        validate_required(anchor_block_root, 'Anchor Block Root')
        validate_type(anchor_block_root, 'Anchor Block Root', str)
        validate_required(anchor_block_number, 'Anchor Block Number')
        validate_type(anchor_block_number, 'Anchor Block Number', int)
        validate_required(beefy_mmr_root, 'Beefy MMR Root')
        validate_type(beefy_mmr_root, 'Beefy MMR Root', str)
        validate_required(current_slot, 'Current Slot')
        validate_type(current_slot, 'Current Slot', int)
        validate_required(current_epoch, 'Current Epoch')
        validate_type(current_epoch, 'Current Epoch', int)
        validate_required(current_guarantors, 'Current Guarantors')
        if not isinstance(current_guarantors, list) or not all(isinstance(g, str) for g in current_guarantors):
            raise ValueError('Current Guarantors must be a list of strings.')
        validate_required(previous_guarantors, 'Previous Guarantors')
        if not isinstance(previous_guarantors, list) or not all(isinstance(g, str) for g in previous_guarantors):
            raise ValueError('Previous Guarantors must be a list of strings.')

        self.anchor_block_root = anchor_block_root
        self.anchor_block_number = anchor_block_number
        self.beefy_mmr_root = beefy_mmr_root
        self.current_slot = current_slot
        self.current_epoch = current_epoch
        self.current_guarantors = current_guarantors
        self.previous_guarantors = previous_guarantors

    def to_object(self) -> dict:
        """
        Converts the RefinementContext to a plain dict for serialization.
        """
        return {
            'anchorBlockRoot': self.anchor_block_root,
            'anchorBlockNumber': self.anchor_block_number,
            'beefyMmrRoot': self.beefy_mmr_root,
            'currentSlot': self.current_slot,
            'currentEpoch': self.current_epoch,
            'currentGuarantors': self.current_guarantors,
            'previousGuarantors': self.previous_guarantors,
        }