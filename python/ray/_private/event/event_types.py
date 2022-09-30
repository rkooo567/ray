from enum import Enum

# Event Source
SOURCE_AUTOSCALER = "AUTOSCALER"
SOURCE_DASHBOARD = "DASHBOARD"


class EventTypes(Enum):
    AUTOSCALER_STARTED = "AUTOSCALER_STARTED"
    AUTOSCALER_EVENT = "AUTOSCALER_EVENT"
    TEST = "TEST"
