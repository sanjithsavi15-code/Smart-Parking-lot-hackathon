"""
env — Smart Parking Environment package.

Public surface:
  from env.models import Car, Slot, Action, Observation, Reward, StepResult, EpisodeSummary
  from env.engine import ParkingEngine, LotGrid, CarQueue, make_car, make_lot, compute_max_revenue
"""

from env.models import (
    Action,
    ActionType,
    Car,
    CarType,
    EpisodeSummary,
    Observation,
    Reward,
    Slot,
    SlotType,
    StepResult,
)
from env.engine import (
    CarQueue,
    LotGrid,
    ParkingEngine,
    compute_max_revenue,
    make_car,
    make_lot,
)

__all__ = [
    # Models
    "Action",
    "ActionType",
    "Car",
    "CarType",
    "EpisodeSummary",
    "Observation",
    "Reward",
    "Slot",
    "SlotType",
    "StepResult",
    # Engine
    "CarQueue",
    "LotGrid",
    "ParkingEngine",
    "compute_max_revenue",
    "make_car",
    "make_lot",
]
