"""
openenv_api.py — OpenEnv-compliant API surface for the Smart Parking Environment.

Public interface:
    env = ParkingEnv()
    obs  = env.reset(task_id="basic_park")
    result = env.step(action)
    obs  = env.state()
    summary = env.summary()

All public methods are stateless from the caller's perspective: they accept and
return fully typed Pydantic models. Internal mutable state lives entirely inside
the ParkingEngine instance.
"""

from __future__ import annotations

from typing import Optional

from env.engine import ParkingEngine
from env.models import (
    Action,
    EpisodeSummary,
    Observation,
    StepResult,
)
from env.tasks import TASK_REGISTRY, TaskConfig, TaskGrader


class ParkingEnv:
    """
    The top-level OpenEnv environment object.

    One instance should be created per session. Call ``reset()`` to start a
    new episode; call ``step()`` repeatedly until ``StepResult.done`` is True;
    call ``summary()`` to retrieve the graded score.
    """

    def __init__(self) -> None:
        self._engine: ParkingEngine = ParkingEngine()
        self._current_task: Optional[TaskConfig] = None
        self._grader: Optional[TaskGrader] = None
        self._episode_active: bool = False
    
    def grade(self):
        """Alias for the OpenEnv validator."""
        return self.summary()    

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, task_id: str) -> Observation:
        """
        Initialise a fresh episode for the given *task_id*.

        Parameters
        ----------
        task_id:
            One of ``"basic_park"``, ``"ev_sort"``, or ``"rush_hour"``.

        Returns
        -------
        Observation
            The initial environment state (before any steps).

        Raises
        ------
        ValueError
            If *task_id* is not registered in the task registry.
        """
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Available tasks: {sorted(TASK_REGISTRY.keys())}"
            )

        task_config: TaskConfig = TASK_REGISTRY[task_id]
        self._current_task = task_config
        self._grader = task_config.grader_cls()

        # Delegate lot/queue construction to the task config
        lot, queue, max_revenue = task_config.build()

        self._engine.load_task(
            task_id=task_id,
            lot=lot,
            queue=queue,
            max_steps=task_config.max_steps,
            max_possible_revenue=max_revenue,
        )
        self._episode_active = True

        return self._engine.current_observation()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Apply *action* to the current environment state and advance by one tick.

        Parameters
        ----------
        action:
            A fully validated ``Action`` instance. Construct via
            ``Action(action_type=ActionType.ASSIGN, car_id=1, slot_id="A1")``.

        Returns
        -------
        StepResult
            Contains the next ``Observation``, shaped ``Reward``, ``done`` flag,
            and diagnostic ``info`` dict.

        Raises
        ------
        RuntimeError
            If ``reset()`` has not been called, or the episode is already done.
        """
        self._assert_active()
        result: StepResult = self._engine.step(action)
        if result.done:
            self._episode_active = False
        return result

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> Observation:
        """
        Return the current environment observation without advancing the clock.

        Safe to call at any point after ``reset()``, including after the episode
        has ended (the terminal state is preserved until the next ``reset()``).

        Raises
        ------
        RuntimeError
            If ``reset()`` has not been called yet.
        """
        self._assert_initialised()
        return self._engine.current_observation()

    # ------------------------------------------------------------------
    # summary()
    # ------------------------------------------------------------------

    def summary(self) -> EpisodeSummary:
        """
        Run the task grader and return a completed ``EpisodeSummary``.

        May be called:
          - After ``StepResult.done == True`` (end-of-episode grading).
          - Mid-episode for the ``rush_hour`` task (continuous grader).

        Raises
        ------
        RuntimeError
            If ``reset()`` has not been called yet.
        """
        self._assert_initialised()
        if self._grader is None or self._current_task is None:
            raise RuntimeError("No active task — call reset() first.")

        score: float = self._grader.grade(self._engine)
        return self._engine.get_summary(final_score=score)

    # ------------------------------------------------------------------
    # Convenience helpers (not part of the core OpenEnv spec)
    # ------------------------------------------------------------------

    @property
    def task_id(self) -> Optional[str]:
        """The task_id of the currently loaded episode, or None."""
        return self._current_task.task_id if self._current_task else None

    @property
    def is_done(self) -> bool:
        return not self._episode_active and self._current_task is not None

    def available_tasks(self) -> list[str]:
        """Return a sorted list of all registered task IDs."""
        return sorted(TASK_REGISTRY.keys())

    # ------------------------------------------------------------------
    # Internal guards
    # ------------------------------------------------------------------

    def _assert_active(self) -> None:
        if not self._episode_active:
            raise RuntimeError(
                "No active episode. Call reset(task_id) to start one."
            )

    def _assert_initialised(self) -> None:
        if self._current_task is None:
            raise RuntimeError(
                "Environment has not been initialised. Call reset(task_id) first."
            )
