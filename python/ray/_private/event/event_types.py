from enum import Enum

# Event Source
SOURCE_AUTOSCALER = "AUTOSCALER"
SOURCE_DASHBOARD = "DASHBOARD"


from enum import Enum


class EventTypes(Enum):
    AUTOSCALER_STARTED = "AUTOSCALER_STARTED"
    AUTOSCALER_EVENT = "AUTOSCALER_EVENT"
    TEST = "TEST"
