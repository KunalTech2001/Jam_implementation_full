from utils.validator import validate_required, validate_type

class WorkItem:
    """
    Represents a Work-Item (W) within a Work-Package.
    Formally defined in Section 14.3 of the graypaper.
    """

    def __init__(self, id: str, program_hash: str, input_data: str, gas_limit: int):
        validate_required(id, 'WorkItem ID')
        validate_type(id, 'WorkItem ID', str)
        validate_required(program_hash, 'WorkItem Program Hash')
        validate_type(program_hash, 'WorkItem Program Hash', str)
        validate_required(input_data, 'WorkItem Input Data')
        validate_type(input_data, 'WorkItem Input Data', str)
        validate_required(gas_limit, 'WorkItem Gas Limit')
        validate_type(gas_limit, 'WorkItem Gas Limit', int)
        if gas_limit <= 0:
            raise ValueError('Gas limit must be positive.')

        self.id = id
        self.program_hash = program_hash
        self.input_data = input_data
        self.gas_limit = gas_limit

    def to_object(self) -> dict:
        """
        Converts the WorkItem to a plain dict for serialization.
        """
        return {
            'id': self.id,
            'programHash': self.program_hash,
            'inputData': self.input_data,
            'gasLimit': self.gas_limit,
        }