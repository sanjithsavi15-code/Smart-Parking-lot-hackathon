"""
models.py — Pydantic V2 typed models for the Smart Parking Environment.

All models used across the OpenEnv API surface: Cars, Slots, Actions,
Observations, Rewards, and Episode metadata.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CarType(str, Enum):
    """The category of an incoming vehicle."""

    STANDARD = "STANDARD"
    EV = "EV"
    VIP = "VIP"


class SlotType(str, Enum):
    """The physical type / capability of a parking slot."""

    STANDARD = "STANDARD"
    EV_CHARGING = "EV_CHARGING"
    PREMIUM = "PREMIUM"


class ActionType(str, Enum):
    """
    The three actions the agent may take on each time-step.

    ASSIGN  — Park a specific car in a specific slot.
    REJECT  — Permanently remove a car from the queue without parking it.
    WAIT    — Do nothing this step (advances the clock by one tick).
    """

    ASSIGN = "ASSIGN"
    REJECT = "REJECT"
    WAIT = "WAIT"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class Car(BaseModel):
    """A vehicle waiting in (or processed from) the incoming queue."""

    id: int = Field(..., description="Unique numeric identifier for this vehicle.")
    car_type: CarType = Field(..., description="Category that determines preferred slot type.")
    entry_time: int = Field(
        ...,
        ge=0,
        description="Simulation time-step at which the car joined the queue.",
    )

    model_config = {"frozen": True}


class Slot(BaseModel):
    """A single parking bay in the lot."""

    id: str = Field(
        ...,
        pattern=r"^[A-Z]\d+$",
        description="Human-readable slot identifier, e.g. 'A1', 'B12'.",
    )
    slot_type: SlotType = Field(..., description="Physical capability of this bay.")
    is_occupied: bool = Field(default=False, description="True when a car is currently parked here.")
    occupant_id: Optional[int] = Field(
        default=None,
        description="ID of the car currently occupying this slot; None when empty.",
    )

    @model_validator(mode="after")
    def occupant_consistency(self) -> "Slot":
        """Ensure occupant_id is set iff is_occupied is True."""
        if self.is_occupied and self.occupant_id is None:
            raise ValueError("is_occupied is True but occupant_id is None.")
        if not self.is_occupied and self.occupant_id is not None:
            raise ValueError("occupant_id is set but is_occupied is False.")
        return self

    model_config = {"frozen": False}  # Mutable so engine can update in-place


# ---------------------------------------------------------------------------
# API surface models
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """
    The full environment state snapshot returned after every step/reset.

    This is the *only* information the agent receives; it must make decisions
    exclusively from these fields.
    """

    available_slots: list[Slot] = Field(
        default_factory=list,
        description="All slots in the lot, regardless of occupancy.",
    )
    incoming_queue: list[Car] = Field(
        default_factory=list,
        description="Ordered list of cars waiting to be assigned or rejected.",
    )
    current_time_step: int = Field(
        default=0,
        ge=0,
        description="Number of steps elapsed since the episode began.",
    )
    current_revenue: float = Field(
        default=0.0,
        ge=0.0,
        description="Cumulative revenue earned so far in this episode.",
    )

    @property
    def free_slots(self) -> list[Slot]:
        """Convenience: returns only unoccupied slots."""
        return [s for s in self.available_slots if not s.is_occupied]

    @property
    def free_slot_ids(self) -> set[str]:
        return {s.id for s in self.free_slots}

    @property
    def queue_car_ids(self) -> set[int]:
        return {c.id for c in self.incoming_queue}


class Action(BaseModel):
    """
    A single decision submitted by the agent to ``step()``.

    Validation rules (enforced at construction time):
    - ASSIGN requires both ``car_id`` and ``slot_id``.
    - REJECT requires ``car_id``; ``slot_id`` must be None.
    - WAIT requires both fields to be None.
    """

    action_type: ActionType = Field(..., description="Which action the agent is taking.")
    car_id: Optional[int] = Field(
        default=None,
        description="ID of the car being acted upon (required for ASSIGN and REJECT).",
    )
    slot_id: Optional[str] = Field(
        default=None,
        description="Target slot ID (required for ASSIGN only).",
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> "Action":
        if self.action_type == ActionType.ASSIGN:
            if self.car_id is None:
                raise ValueError("ASSIGN action requires car_id.")
            if self.slot_id is None:
                raise ValueError("ASSIGN action requires slot_id.")
        elif self.action_type == ActionType.REJECT:
            if self.car_id is None:
                raise ValueError("REJECT action requires car_id.")
            if self.slot_id is not None:
                raise ValueError("REJECT action must not specify slot_id.")
        elif self.action_type == ActionType.WAIT:
            if self.car_id is not None or self.slot_id is not None:
                raise ValueError("WAIT action must have car_id=None and slot_id=None.")
        return self

    model_config = {"frozen": True}


class Reward(BaseModel):
    """
    The immediate shaped reward returned alongside each Observation.

    ``value``    — The scalar reward for this step (can be negative).
    ``breakdown``— Human-readable dict explaining which components fired.
    """

    value: float = Field(..., description="Scalar reward signal for this time-step.")
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-component reward breakdown for interpretability.",
    )

    model_config = {"frozen": True}


class StepResult(BaseModel):
    """The complete return value of a single ``step()`` call."""

    observation: Observation
    reward: Reward
    done: bool = Field(
        default=False,
        description="True when the episode has ended (queue empty or step limit reached).",
    )
    info: dict[str, object] = Field(
        default_factory=dict,
        description="Auxiliary diagnostics; not used by the agent for decisions.",
    )

    model_config = {"frozen": True}


class EpisodeSummary(BaseModel):
    """
    Returned by ``env.summary()`` at the end of an episode.

    Used by the task graders in tasks.py to compute the final score.
    """

    task_id: str
    total_steps: int
    total_revenue: float
    max_possible_revenue: float
    cars_parked: int
    cars_rejected: int
    invalid_actions: int
    final_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised score in [0, 1] as computed by the task grader.",
    )

    model_config = {"frozen": True}
