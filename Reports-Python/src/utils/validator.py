class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_required(value, field_name: str) -> None:
    """
    Validates that a required value is present.
    """
    if value is None or (isinstance(value, str) and value.strip() == ''):
        raise ValidationError(f"{field_name} is required.")

def validate_type(value, field_name: str, expected_type: type) -> None:
    """
    Validates that a value matches the expected primitive type.
    """
    if not isinstance(value, expected_type):
        raise ValidationError(f"{field_name} must be a {expected_type.__name__}.")

def validate_array_of_type(arr, field_name: str, expected_type: type) -> None:
    """
    Validates that all items in an array are of the expected type.
    """
    if not isinstance(arr, list):
        raise ValidationError(f"{field_name} must be a list.")
    for item in arr:
        if not isinstance(item, expected_type):
            raise ValidationError(f"All items in {field_name} must be a {expected_type.__name__}.")

def validate_instance_of(value, field_name: str, expected_class: type) -> None:
    """
    Validates that a value is an instance of a given class.
    """
    if not isinstance(value, expected_class):
        raise ValidationError(f"{field_name} must be an instance of {expected_class.__name__}.")