"""Model-agnostic contracts for the KineWorld Research Harness.

The contract deliberately treats tensors and simulator-specific objects as opaque
payloads.  Adapters own conversion; the harness owns protocol, accounting, and
evidence.  Adding an adapter does not imply that the underlying capability has
been validated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class Capability(str, Enum):
    ENCODE = "encode"
    BELIEF_UPDATE = "belief_update"
    PREDICT = "predict"
    UNCERTAINTY = "uncertainty"
    PLAN = "plan"
    ADAPT = "adapt"


class CapabilityNotSupported(NotImplementedError):
    """Raised when a protocol requests a capability the adapter did not declare."""


@dataclass(frozen=True)
class AdapterMetadata:
    name: str
    version: str
    implementation_source: str
    source_commit: str
    model_hash_sha256: str | None
    license_id: str
    capabilities: frozenset[Capability]
    third_party: bool = True
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        required = {
            "name": self.name,
            "version": self.version,
            "implementation_source": self.implementation_source,
            "source_commit": self.source_commit,
            "license_id": self.license_id,
        }
        missing = [key for key, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(f"adapter metadata missing: {', '.join(missing)}")
        if self.model_hash_sha256 is not None:
            digest = self.model_hash_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("model_hash_sha256 must be a 64-character hex digest")


@dataclass(frozen=True)
class Observation:
    payload: Any
    timestamp_s: float | None = None
    modalities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncodedObservation:
    latent: Any
    representation_size: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeliefState:
    state: Any
    step: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSequence:
    actions: Sequence[Any]
    action_block: int = 1

    def __post_init__(self) -> None:
        if self.action_block < 1:
            raise ValueError("action_block must be >= 1")
        if not self.actions:
            raise ValueError("actions must not be empty")


@dataclass(frozen=True)
class Prediction:
    future: Any
    horizon: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UncertaintyEstimate:
    value: Any
    method: str
    calibrated: bool
    coverage_target: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlanBudget:
    horizon: int
    max_model_evaluations: int
    max_wall_time_s: float | None = None

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.max_model_evaluations < 1:
            raise ValueError("plan budget values must be positive")


@dataclass(frozen=True)
class PlanResult:
    actions: ActionSequence
    objective: float | None
    model_evaluations: int
    latency_s: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class WorldModelAdapter(ABC):
    """Minimum adapter boundary; algorithms remain in their official codebases."""

    def __init__(self, metadata: AdapterMetadata) -> None:
        metadata.validate()
        self._metadata = metadata

    @property
    def metadata(self) -> AdapterMetadata:
        return self._metadata

    def require(self, capability: Capability) -> None:
        if capability not in self.metadata.capabilities:
            raise CapabilityNotSupported(
                f"{self.metadata.name} does not declare capability '{capability.value}'"
            )

    @abstractmethod
    def encode(self, observation: Observation) -> EncodedObservation:
        """Map an observation into the model's internal representation."""

    @abstractmethod
    def update_belief(
        self,
        previous: BeliefState | None,
        observation: Observation,
        previous_action: Any | None,
    ) -> BeliefState:
        """Update persistent belief; memoryless models must declare that in metadata."""

    @abstractmethod
    def predict(self, belief: BeliefState, actions: ActionSequence) -> Prediction:
        """Predict action-conditioned future state or representation."""

    def estimate_uncertainty(self, prediction: Prediction) -> UncertaintyEstimate:
        self.require(Capability.UNCERTAINTY)
        raise CapabilityNotSupported(
            f"{self.metadata.name} declared uncertainty but did not implement it"
        )

    @abstractmethod
    def plan(self, belief: BeliefState, goal: Any, budget: PlanBudget) -> PlanResult:
        """Plan under an explicit, auditable compute budget."""

    def adapt(self, transition: Mapping[str, Any]) -> Mapping[str, Any]:
        self.require(Capability.ADAPT)
        raise CapabilityNotSupported(
            f"{self.metadata.name} declared adaptation but did not implement it"
        )
