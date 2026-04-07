"""
engine.py — Core simulation engine for the Smart Parking Environment.

Responsibilities:
  - Maintain the authoritative lot state (slots + parked cars).
  - Maintain the incoming car queue.
  - Execute validated actions and mutate state accordingly.
  - Compute per-step shaped rewards.
  - Track episode-level statistics for the task graders.

This module is intentionally side-effect-free with respect to I/O; all
persistence and API routing is handled by openenv_api.py above it.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional

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

# ---------------------------------------------------------------------------
# Revenue table — how much each (car_type, slot_type) assignment earns
# ---------------------------------------------------------------------------

REVENUE_TABLE: dict[tuple[CarType, SlotType], float] = {
    # Standard car
    (CarType.STANDARD, SlotType.STANDARD): 10.0,
    (CarType.STANDARD, SlotType.EV_CHARGING): 0.0,   # Not allowed → penalty
    (CarType.STANDARD, SlotType.PREMIUM): 10.0,       # Allowed but wastes premium
    # EV
    (CarType.EV, SlotType.STANDARD): 10.0,            # Allowed but suboptimal
    (CarType.EV, SlotType.EV_CHARGING): 15.0,         # Optimal
    (CarType.EV, SlotType.PREMIUM): 10.0,
    # VIP
    (CarType.VIP, SlotType.STANDARD): 10.0,           # Allowed but suboptimal
    (CarType.VIP, SlotType.EV_CHARGING): 0.0,         # Not allowed → penalty
    (CarType.VIP, SlotType.PREMIUM): 20.0,            # Optimal
}

# Maximum revenue per car if optimally placed (used for normalisation)
MAX_REVENUE_PER_CAR: dict[CarType, float] = {
    CarType.STANDARD: 10.0,
    CarType.EV: 15.0,
    CarType.VIP: 20.0,
}

# ---------------------------------------------------------------------------
# Reward constants (mirrors the spec in openenv_api.py)
# ---------------------------------------------------------------------------

REWARD_PARK_BASE: float = 0.2
REWARD_TYPE_MATCH_BONUS: float = 0.5
PENALTY_REJECT_VALID: float = -0.1
PENALTY_INVALID_ACTION: float = -0.5


# ---------------------------------------------------------------------------
# Helper: slot-type compatibility check
# ---------------------------------------------------------------------------

def _is_assignment_valid(car: Car, slot: Slot) -> bool:
    """
    Returns True when placing *car* in *slot* is a legal (non-penalised) move.

    Rules:
      - STANDARD cars may NOT occupy EV_CHARGING slots.
      - VIP cars may NOT occupy EV_CHARGING slots.
      - All other combinations are legal (though some earn a type-match bonus).
    """
    forbidden: set[tuple[CarType, SlotType]] = {
        (CarType.STANDARD, SlotType.EV_CHARGING),
        (CarType.VIP, SlotType.EV_CHARGING),
    }
    return (car.car_type, slot.slot_type) not in forbidden


def _is_type_match(car: Car, slot: Slot) -> bool:
    """Returns True when the assignment earns the +0.5 type-match bonus."""
    matches: set[tuple[CarType, SlotType]] = {
        (CarType.EV, SlotType.EV_CHARGING),
        (CarType.VIP, SlotType.PREMIUM),
    }
    return (car.car_type, slot.slot_type) in matches


# ---------------------------------------------------------------------------
# LotGrid — the physical parking lot
# ---------------------------------------------------------------------------

@dataclass
class LotGrid:
    """
    Represents the full set of parking slots.

    Slots are stored in an ordered dict keyed by slot_id for O(1) lookup.
    """

    slots: dict[str, Slot] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        n_standard: int,
        n_ev: int,
        n_premium: int,
        rng: Optional[random.Random] = None,
    ) -> "LotGrid":
        """
        Construct a fresh lot with the requested slot counts.

        Row labels: A=standard, B=EV charging, C=premium.
        """
        slots: dict[str, Slot] = {}

        def _add(prefix: str, count: int, slot_type: SlotType) -> None:
            for i in range(1, count + 1):
                sid = f"{prefix}{i}"
                slots[sid] = Slot(id=sid, slot_type=slot_type, is_occupied=False, occupant_id=None)

        _add("A", n_standard, SlotType.STANDARD)
        _add("B", n_ev, SlotType.EV_CHARGING)
        _add("C", n_premium, SlotType.PREMIUM)
        return cls(slots=slots)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_slot(self, slot_id: str) -> Optional[Slot]:
        return self.slots.get(slot_id)

    def all_slots(self) -> list[Slot]:
        return list(self.slots.values())

    def free_slots(self) -> list[Slot]:
        return [s for s in self.slots.values() if not s.is_occupied]

    def occupied_slots(self) -> list[Slot]:
        return [s for s in self.slots.values() if s.is_occupied]

    def has_any_free_slot(self) -> bool:
        return any(not s.is_occupied for s in self.slots.values())

    def has_compatible_free_slot(self, car: Car) -> bool:
        """True if at least one free slot would be a *valid* (non-penalised) match."""
        return any(
            not s.is_occupied and _is_assignment_valid(car, s)
            for s in self.slots.values()
        )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def park(self, car: Car, slot_id: str) -> None:
        """Mark the slot as occupied. Caller must validate beforehand."""
        slot = self.slots[slot_id]
        slot.is_occupied = True
        slot.occupant_id = car.id

    def vacate(self, slot_id: str) -> None:
        """Free a slot (used in rush_hour pre-fill teardown if needed)."""
        slot = self.slots[slot_id]
        slot.is_occupied = False
        slot.occupant_id = None

    def snapshot(self) -> list[Slot]:
        """Return a deep copy of all slots (safe for inclusion in Observation)."""
        return [copy.deepcopy(s) for s in self.slots.values()]


# ---------------------------------------------------------------------------
# CarQueue — the FIFO queue of incoming vehicles
# ---------------------------------------------------------------------------

@dataclass
class CarQueue:
    cars: list[Car] = field(default_factory=list)

    def enqueue(self, car: Car) -> None:
        self.cars.append(car)

    def remove(self, car_id: int) -> Optional[Car]:
        for i, c in enumerate(self.cars):
            if c.id == car_id:
                return self.cars.pop(i)
        return None

    def get(self, car_id: int) -> Optional[Car]:
        for c in self.cars:
            if c.id == car_id:
                return c
        return None

    def is_empty(self) -> bool:
        return len(self.cars) == 0

    def snapshot(self) -> list[Car]:
        return list(self.cars)  # Cars are frozen; shallow copy is safe


# ---------------------------------------------------------------------------
# ParkingEngine — the main simulation controller
# ---------------------------------------------------------------------------

class ParkingEngine:
    """
    Orchestrates the parking lot simulation for a single episode.

    Lifecycle:
        engine = ParkingEngine()
        engine.load_task(task_config)   # called by openenv_api.reset()
        while not done:
            result = engine.step(action)
        summary = engine.get_summary()
    """

    # ------------------------------------------------------------------
    # Construction & task loading
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._lot: LotGrid = LotGrid()
        self._queue: CarQueue = CarQueue()
        self._time_step: int = 0
        self._max_steps: int = 50
        self._revenue: float = 0.0
        self._max_possible_revenue: float = 0.0
        self._task_id: str = ""
        self._done: bool = False

        # Episode-level counters for the grader
        self._cars_parked: int = 0
        self._cars_rejected: int = 0
        self._invalid_actions: int = 0

        # Map car_id → Car for cars that have been parked (for grader queries)
        self._parked_cars: dict[int, Car] = {}
        # Map car_id → slot_id for grader
        self._parking_map: dict[int, str] = {}

    def load_task(
        self,
        task_id: str,
        lot: LotGrid,
        queue: CarQueue,
        max_steps: int,
        max_possible_revenue: float,
    ) -> None:
        """Initialise engine state from a task definition."""
        self._task_id = task_id
        self._lot = lot
        self._queue = queue
        self._max_steps = max_steps
        self._max_possible_revenue = max_possible_revenue
        self._time_step = 0
        self._revenue = 0.0
        self._done = False
        self._cars_parked = 0
        self._cars_rejected = 0
        self._invalid_actions = 0
        self._parked_cars = {}
        self._parking_map = {}

    # ------------------------------------------------------------------
    # Core step logic
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Apply *action* to the current environment state.

        Returns a StepResult containing:
          - The next Observation.
          - A shaped Reward.
          - A ``done`` flag.
          - An ``info`` dict with diagnostic details.
        """
        if self._done:
            raise RuntimeError(
                "step() called on a finished episode. Call reset() first."
            )

        reward_components: dict[str, float] = {}
        info: dict[str, object] = {}

        # ----------------------------------------------------------------
        # Dispatch on action type
        # ----------------------------------------------------------------

        if action.action_type == ActionType.ASSIGN:
            reward_components, info = self._handle_assign(action)

        elif action.action_type == ActionType.REJECT:
            reward_components, info = self._handle_reject(action)

        else:  # WAIT
            reward_components = {"wait": 0.0}
            info = {"message": "Agent chose to wait."}

        # ----------------------------------------------------------------
        # Advance clock
        # ----------------------------------------------------------------

        self._time_step += 1

        # ----------------------------------------------------------------
        # Check episode termination
        # ----------------------------------------------------------------

        episode_done = self._queue.is_empty() or self._time_step >= self._max_steps

        # Terminal revenue-normalisation bonus
        if episode_done:
            terminal_bonus = (
                self._revenue / self._max_possible_revenue
                if self._max_possible_revenue > 0
                else 0.0
            )
            reward_components["terminal_revenue_ratio"] = terminal_bonus
            self._done = True

        total_reward = sum(reward_components.values())
        reward = Reward(value=total_reward, breakdown=reward_components)

        obs = self._build_observation()

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info=info,
        )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_assign(
        self, action: Action
    ) -> tuple[dict[str, float], dict[str, object]]:
        """Process an ASSIGN action. Returns (reward_breakdown, info)."""
        components: dict[str, float] = {}
        car_id: int = action.car_id  # type: ignore[assignment]
        slot_id: str = action.slot_id  # type: ignore[assignment]

        # --- Validate car is in queue ---
        car = self._queue.get(car_id)
        if car is None:
            self._invalid_actions += 1
            components["invalid_car_not_in_queue"] = PENALTY_INVALID_ACTION
            return components, {
                "error": f"Car {car_id} is not in the incoming queue.",
                "action": "ASSIGN",
            }

        # --- Validate slot exists ---
        slot = self._lot.get_slot(slot_id)
        if slot is None:
            self._invalid_actions += 1
            components["invalid_slot_not_found"] = PENALTY_INVALID_ACTION
            return components, {
                "error": f"Slot {slot_id} does not exist.",
                "action": "ASSIGN",
            }

        # --- Validate slot is free ---
        if slot.is_occupied:
            self._invalid_actions += 1
            components["invalid_slot_occupied"] = PENALTY_INVALID_ACTION
            return components, {
                "error": f"Slot {slot_id} is already occupied.",
                "action": "ASSIGN",
            }

        # --- Validate car/slot type compatibility ---
        if not _is_assignment_valid(car, slot):
            self._invalid_actions += 1
            components["invalid_type_mismatch"] = PENALTY_INVALID_ACTION
            return components, {
                "error": (
                    f"Car type {car.car_type} cannot be placed in "
                    f"slot type {slot.slot_type}."
                ),
                "action": "ASSIGN",
            }

        # --- Execute the parking ---
        self._queue.remove(car_id)
        self._lot.park(car, slot_id)
        revenue_earned = REVENUE_TABLE[(car.car_type, slot.slot_type)]
        self._revenue += revenue_earned
        self._cars_parked += 1
        self._parked_cars[car.id] = car
        self._parking_map[car.id] = slot_id

        # Reward
        components["park_base"] = REWARD_PARK_BASE
        if _is_type_match(car, slot):
            components["type_match_bonus"] = REWARD_TYPE_MATCH_BONUS

        return components, {
            "message": f"Car {car_id} ({car.car_type}) parked in {slot_id} ({slot.slot_type}). Revenue: +{revenue_earned}.",
            "revenue_earned": revenue_earned,
            "action": "ASSIGN",
        }

    def _handle_reject(
        self, action: Action
    ) -> tuple[dict[str, float], dict[str, object]]:
        """Process a REJECT action. Returns (reward_breakdown, info)."""
        components: dict[str, float] = {}
        car_id: int = action.car_id  # type: ignore[assignment]

        car = self._queue.get(car_id)
        if car is None:
            self._invalid_actions += 1
            components["invalid_car_not_in_queue"] = PENALTY_INVALID_ACTION
            return components, {
                "error": f"Car {car_id} not in queue — cannot reject.",
                "action": "REJECT",
            }

        # Apply penalty if valid slots existed for this car
        had_valid_slot = self._lot.has_compatible_free_slot(car)
        self._queue.remove(car_id)
        self._cars_rejected += 1

        if had_valid_slot:
            components["reject_with_valid_slot"] = PENALTY_REJECT_VALID
            msg = (
                f"Car {car_id} rejected despite valid slots being available. Penalty applied."
            )
        else:
            components["reject_no_valid_slot"] = 0.0
            msg = f"Car {car_id} rejected; no valid slots were available."

        return components, {"message": msg, "action": "REJECT"}

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        return Observation(
            available_slots=self._lot.snapshot(),
            incoming_queue=self._queue.snapshot(),
            current_time_step=self._time_step,
            current_revenue=self._revenue,
        )

    # ------------------------------------------------------------------
    # Public read-only accessors (used by openenv_api and tasks)
    # ------------------------------------------------------------------

    def current_observation(self) -> Observation:
        return self._build_observation()

    def is_done(self) -> bool:
        return self._done

    @property
    def time_step(self) -> int:
        return self._time_step

    @property
    def revenue(self) -> float:
        return self._revenue

    @property
    def max_possible_revenue(self) -> float:
        return self._max_possible_revenue

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def parked_cars(self) -> dict[int, Car]:
        """Dict of car_id → Car for all cars successfully parked this episode."""
        return dict(self._parked_cars)

    @property
    def parking_map(self) -> dict[int, str]:
        """Dict of car_id → slot_id for all successfully parked cars."""
        return dict(self._parking_map)

    @property
    def lot(self) -> LotGrid:
        return self._lot

    @property
    def queue(self) -> CarQueue:
        return self._queue

    # ------------------------------------------------------------------
    # Summary (for graders)
    # ------------------------------------------------------------------

    def get_summary(self, final_score: float) -> EpisodeSummary:
        return EpisodeSummary(
            task_id=self._task_id,
            total_steps=self._time_step,
            total_revenue=self._revenue,
            max_possible_revenue=self._max_possible_revenue,
            cars_parked=self._cars_parked,
            cars_rejected=self._cars_rejected,
            invalid_actions=self._invalid_actions,
            final_score=final_score,
        )


# ---------------------------------------------------------------------------
# Convenience factory functions (used by tasks.py)
# ---------------------------------------------------------------------------

def make_car(
    car_id: int,
    car_type: CarType,
    entry_time: int = 0,
) -> Car:
    return Car(id=car_id, car_type=car_type, entry_time=entry_time)


def make_lot(
    n_standard: int = 5,
    n_ev: int = 3,
    n_premium: int = 2,
    rng: Optional[random.Random] = None,
) -> LotGrid:
    return LotGrid.build(n_standard=n_standard, n_ev=n_ev, n_premium=n_premium, rng=rng)


def compute_max_revenue(cars: list[Car], lot: LotGrid) -> float:
    """
    Compute the theoretical maximum revenue if every car were optimally placed.

    This is an upper-bound estimate: it assumes unlimited optimal slots exist.
    Used for terminal reward normalisation and grader scoring.
    """
    return sum(MAX_REVENUE_PER_CAR[car.car_type] for car in cars)
