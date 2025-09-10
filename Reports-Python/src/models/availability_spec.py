from utils.validator import validate_required, validate_type

class AvailabilitySpec:
    """
    Represents the Availability Specification (Y) for a Work-Report.
    Defines how the report data is packaged for availability.
    """

    def __init__(self, total_fragments: int, data_fragments: int, fragment_hashes: list[str]):
        
        validate_required(total_fragments, 'Total Fragments')
        validate_type(total_fragments, 'Total Fragments', int)
        if total_fragments <= 0:
            raise ValueError('Total fragments must be positive.')

        validate_required(data_fragments, 'Data Fragments')
        validate_type(data_fragments, 'Data Fragments', int)
        if data_fragments <= 0 or data_fragments > total_fragments:
            raise ValueError('Data fragments must be positive and less than or equal to total fragments.')

        validate_required(fragment_hashes, 'Fragment Hashes')
        if (not isinstance(fragment_hashes, list) or
            len(fragment_hashes) != total_fragments or
            not all(isinstance(h, str) for h in fragment_hashes)):
            raise ValueError('Fragment Hashes must be a list of strings matching total_fragments length.')

        self.total_fragments = total_fragments
        self.data_fragments = data_fragments
        self.fragment_hashes = fragment_hashes

    def to_object(self) -> dict:
        """
        Converts the AvailabilitySpec to a plain dict for serialization.
        """
        return {
            'totalFragments': self.total_fragments,
            'dataFragments': self.data_fragments,
            'fragmentHashes': self.fragment_hashes,
        }