from utils.validator import (
    validate_required,
    validate_type,
    validate_array_of_type,
    validate_instance_of,
)
from models.work_item import WorkItem

class WorkPackage:
    """
    Represents a Work-Package (P), the atomic unit of intent.
    Formally defined in Section 14.3 of the graypaper.
    """

    def __init__(
        self,
        authorization_token: str,
        authorization_service_details: dict,
        context: str,
        work_items: list[WorkItem]
    ):
        validate_required(authorization_token, 'Authorization Token')
        validate_type(authorization_token, 'Authorization Token', str)

        validate_required(authorization_service_details, 'Authorization Service Details')
        validate_type(authorization_service_details, 'Authorization Service Details', dict)
        validate_required(authorization_service_details.get('h'), 'Auth Service Host')
        validate_type(authorization_service_details.get('h'), 'Auth Service Host', str)
        validate_required(authorization_service_details.get('u'), 'Auth Service URL')
        validate_type(authorization_service_details.get('u'), 'Auth Service URL', str)
        validate_required(authorization_service_details.get('f'), 'Auth Service Function')
        validate_type(authorization_service_details.get('f'), 'Auth Service Function', str)

        validate_required(context, 'Context')
        validate_type(context, 'Context', str)

        validate_array_of_type(work_items, 'Work Items', WorkItem)
        if len(work_items) == 0:
            raise ValueError('Work-Package must contain at least one Work-Item.')
        for item in work_items:
            validate_instance_of(item, 'Work Item', WorkItem)

        self.authorization_token = authorization_token
        self.authorization_service_details = authorization_service_details
        self.context = context
        self.work_items = work_items

    def to_object(self) -> dict:
        """
        Converts the WorkPackage to a plain dict for serialization.
        """
        return {
            'authorizationToken': self.authorization_token,
            'authorizationServiceDetails': self.authorization_service_details,
            'context': self.context,
            'workItems': [item.to_object() for item in self.work_items],
        }