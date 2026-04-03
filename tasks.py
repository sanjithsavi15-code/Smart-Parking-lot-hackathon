"""
tasks.py — Task definitions and deterministic graders for the Smart Parking Environment.

Each task is described by a ``TaskConfig`` dataclass which:
  1. Defines the lot layout and initial car queue (via ``build()``).
  2. Points to a ``TaskGrader`` subclass that scores the episode.

Task registry:
  TASK_REGISTRY["basic_park"]  → Task 1 (Easy)
  TASK_REGISTRY["ev_sort"]     → Task 2 (Medium)
  TASK_REGISTRY["rush_hour"]   → Task 3 (Hard)

Graders are pure functions of ``ParkingEngine`` state — no side effects.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type

from env.engine import (
    CarQueue,
    LotGrid,
    ParkingEngine,
    compute_max_revenue,
    make_car,
    make_lot,
)
from env.models import Car, CarType, SlotType

if TYPE_CHECKING:
    pass  # avoid circular imports; ParkingEngine imported at runtime only


# ---------------------------------------------------------------------------
# Abstract base grader
# ---------------------------------------------------------------------------


class TaskGrader(abc.ABC):
    """
    Base class for all task graders.

    ``grade()`` receives the live ``ParkingEngine`` instance and returns a
    normalised score in [0.0, 1.0].  It must be a pure read-only operation.
    """

    @abc.abstractmethod
    def grade(self, engine: ParkingEngine) -> float: ...


# ---------------------------------------------------------------------------
# Task configuration container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskConfig:
    """
    Immutable description of a single task scenario.

    Attributes
    ----------
    task_id:
        Unique string identifier used in ``reset(task_id=...)``.
    display_name:
        Human-readable name shown in the UI and README.
    difficulty:
        "Easy" | "Medium" | "Hard"
    description:
        One-sentence summary for the openenv.yaml / README.
    max_steps:
        Episode terminates after this many steps regardless of queue size.
    grader_cls:
        The ``TaskGrader`` subclass that scores this task.
    _builder:
        Private callable that constructs (LotGrid, CarQueue, max_revenue).
        Stored as a field with a leading underscore to signal it is internal.
    """

    task_id: str
    display_name: str
    difficulty: str
    description: str
    max_steps: int
    grader_cls: Type[TaskGrader]
    _builder: object = field(repr=False)  # Callable[[], tuple[LotGrid, CarQueue, float]]

    def build(self) -> tuple[LotGrid, CarQueue, float]:
        """Construct and return a fresh (lot, queue, max_revenue) triple."""
        return self._builder()  # type: ignore[operator]


# ===========================================================================
# Task 1 — basic_park (Easy)
# ===========================================================================
#
# Scenario  : 5 standard cars, empty lot with 8 standard + 3 EV + 2 premium slots.
# Goal      : Park every car (queue must be empty, all 5 in valid slots).
# Max steps : 20  (generous; this is a warm-up task)
# ===========================================================================


class BasicParkGrader(TaskGrader):
    """
    Returns 1.0 iff:
      - The incoming queue is empty (no cars left unprocessed), AND
      - Exactly 5 cars are parked in valid (non-EV) slots.

    Partial credit: 0.2 per car correctly parked (for mid-episode snapshots).
    """

    EXPECTED_CARS: int = 5

    def grade(self, engine: ParkingEngine) -> float:
        queue_empty = engine.queue.is_empty()
        parked_count = len(engine.parked_cars)

        # Full score: all cars parked, queue empty
        if queue_empty and parked_count == self.EXPECTED_CARS:
            return 1.0

        # Partial credit: proportion of cars successfully parked
        partial = parked_count / self.EXPECTED_CARS
        return round(min(partial, 1.0), 4)


def _build_basic_park() -> tuple[LotGrid, CarQueue, float]:
    lot = make_lot(n_standard=8, n_ev=3, n_premium=2)
    cars: list[Car] = [
        make_car(car_id=i, car_type=CarType.STANDARD, entry_time=0)
        for i in range(1, 6)
    ]
    queue = CarQueue()
    for c in cars:
        queue.enqueue(c)
    max_revenue = compute_max_revenue(cars, lot)
    return lot, queue, max_revenue


# ===========================================================================
# Task 2 — ev_sort (Medium)
# ===========================================================================
#
# Scenario  : 6-car mixed queue (3 EV + 3 Standard). Only 3 EV_CHARGING slots,
#             4 standard slots, 2 premium slots.
# Goal      : All EVs in EV_CHARGING; all Standard cars in STANDARD slots.
#             Premium slots exist as a "temptation" / distractor.
# Max steps : 20
# ===========================================================================


class EvSortGrader(TaskGrader):
    """
    Returns 1.0 ONLY if:
      - Every EV car is parked in an EV_CHARGING slot.
      - Every Standard car is parked in a STANDARD slot.
      - The queue is empty (all cars processed).

    Partial scoring breakdown:
      - 1/3 weight per EV correctly placed (max 1.0 from 3 EVs × 1/3).
      - 1/3 weight per Standard correctly placed (max 1.0 from 3 std × 1/3).
      - Final score = (correctly_placed / total_cars).
    """

    EV_IDS: frozenset[int] = frozenset({1, 2, 3})       # EV cars
    STD_IDS: frozenset[int] = frozenset({4, 5, 6})      # Standard cars

    def grade(self, engine: ParkingEngine) -> float:
        parking_map = engine.parking_map           # car_id → slot_id
        parked_cars = engine.parked_cars            # car_id → Car
        lot = engine.lot

        correctly_placed = 0
        total = len(self.EV_IDS) + len(self.STD_IDS)

        for car_id, slot_id in parking_map.items():
            car = parked_cars.get(car_id)
            slot = lot.get_slot(slot_id)
            if car is None or slot is None:
                continue

            if car.car_type == CarType.EV and slot.slot_type == SlotType.EV_CHARGING:
                correctly_placed += 1
            elif car.car_type == CarType.STANDARD and slot.slot_type == SlotType.STANDARD:
                correctly_placed += 1

        queue_empty = engine.queue.is_empty()

        # Perfect score requires all cars correctly placed AND queue empty
        if correctly_placed == total and queue_empty:
            return 1.0

        return round(correctly_placed / total, 4)


def _build_ev_sort() -> tuple[LotGrid, CarQueue, float]:
    # Tight EV supply: exactly 3 EV_CHARGING slots for 3 EV cars
    lot = make_lot(n_standard=4, n_ev=3, n_premium=2)
    cars: list[Car] = [
        make_car(car_id=1, car_type=CarType.EV, entry_time=0),
        make_car(car_id=2, car_type=CarType.EV, entry_time=0),
        make_car(car_id=3, car_type=CarType.EV, entry_time=0),
        make_car(car_id=4, car_type=CarType.STANDARD, entry_time=0),
        make_car(car_id=5, car_type=CarType.STANDARD, entry_time=0),
        make_car(car_id=6, car_type=CarType.STANDARD, entry_time=0),
    ]
    queue = CarQueue()
    for c in cars:
        queue.enqueue(c)
    max_revenue = compute_max_revenue(cars, lot)
    return lot, queue, max_revenue


# ===========================================================================
# Task 3 — rush_hour (Hard)
# ===========================================================================
#
# Scenario  : Lot is ~90% full (9 of 10 standard slots occupied, all 3 EV
#             occupied, 1 of 2 premium occupied).
#             Queue: 3 VIP + 4 Standard cars.
#             Agent must maximise revenue within 15 steps.
# Goal      : Continuous score = current_revenue / theoretical_max_revenue.
# Max steps : 15
#
# Design intent: The agent must decide which cars to park in the 2 remaining
# free slots (1 STANDARD + 1 PREMIUM) and which to reject — all under pressure.
# ===========================================================================


class RushHourGrader(TaskGrader):
    """
    Continuous revenue-optimisation grader.

    Score = engine.revenue / engine.max_possible_revenue

    This is a real-valued score in [0.0, 1.0].  An agent that parks both VIPs
    in the premium slot (only one is free) and the best-paying remaining car
    will approach 1.0.
    """

    def grade(self, engine: ParkingEngine) -> float:
        if engine.max_possible_revenue <= 0:
            return 0.0
        raw = engine.revenue / engine.max_possible_revenue
        return round(min(max(raw, 0.0), 1.0), 4)


def _build_rush_hour() -> tuple[LotGrid, CarQueue, float]:
    """
    Build a nearly-full lot and a high-pressure queue.

    Lot layout (13 slots total):
      A1–A10 : STANDARD  (9 occupied, 1 free → A10)
      B1–B3  : EV_CHARGING (all 3 occupied)
      C1–C2  : PREMIUM  (C1 occupied, C2 free)

    Queue (7 cars):
      VIP   : ids 101, 102, 103
      STD   : ids 201, 202, 203, 204
    """
    lot = make_lot(n_standard=10, n_ev=3, n_premium=2)

    # --- Pre-fill the lot to ~90% occupancy ---
    # Standard slots A1–A9 occupied by phantom cars (ids 900–908)
    for i in range(1, 10):
        slot_id = f"A{i}"
        slot = lot.get_slot(slot_id)
        if slot:
            slot.is_occupied = True
            slot.occupant_id = 900 + i

    # All EV slots occupied
    for i in range(1, 4):
        slot_id = f"B{i}"
        slot = lot.get_slot(slot_id)
        if slot:
            slot.is_occupied = True
            slot.occupant_id = 910 + i

    # Premium slot C1 occupied, C2 remains free
    c1 = lot.get_slot("C1")
    if c1:
        c1.is_occupied = True
        c1.occupant_id = 920

    # --- Incoming queue ---
    incoming: list[Car] = [
        make_car(car_id=101, car_type=CarType.VIP, entry_time=0),
        make_car(car_id=102, car_type=CarType.VIP, entry_time=0),
        make_car(car_id=103, car_type=CarType.VIP, entry_time=0),
        make_car(car_id=201, car_type=CarType.STANDARD, entry_time=0),
        make_car(car_id=202, car_type=CarType.STANDARD, entry_time=0),
        make_car(car_id=203, car_type=CarType.STANDARD, entry_time=0),
        make_car(car_id=204, car_type=CarType.STANDARD, entry_time=0),
    ]
    queue = CarQueue()
    for c in incoming:
        queue.enqueue(c)

    # Max possible revenue = best assignment for all 7 cars if slots existed
    max_revenue = compute_max_revenue(incoming, lot)
    return lot, queue, max_revenue


# ===========================================================================
# Task registry — single source of truth
# ===========================================================================

TASK_REGISTRY: dict[str, TaskConfig] = {
    "basic_park": TaskConfig(
        task_id="basic_park",
        display_name="Basic Park",
        difficulty="Easy",
        description=(
            "Park 5 standard cars in an empty lot. "
            "Score 1.0 if all 5 are parked and the queue is empty."
        ),
        max_steps=20,
        grader_cls=BasicParkGrader,
        _builder=_build_basic_park,
    ),
    "ev_sort": TaskConfig(
        task_id="ev_sort",
        display_name="EV Sort",
        difficulty="Medium",
        description=(
            "Sort 3 EVs into EV_CHARGING slots and 3 Standard cars into Standard slots. "
            "Score 1.0 only if every car is in its optimal slot type."
        ),
        max_steps=20,
        grader_cls=EvSortGrader,
        _builder=_build_ev_sort,
    ),
    "rush_hour": TaskConfig(
        task_id="rush_hour",
        display_name="Rush Hour",
        difficulty="Hard",
        description=(
            "Lot is 90% full. Maximise revenue from a mixed VIP/Standard queue "
            "within 15 steps. Score = revenue / theoretical_max_revenue."
        ),
        max_steps=15,
        grader_cls=RushHourGrader,
        _builder=_build_rush_hour,
    ),
}
