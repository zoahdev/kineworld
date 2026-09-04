from __future__ import annotations

import unittest

from kineworld.harness import (
    ActionSequence,
    AdapterMetadata,
    BeliefState,
    Capability,
    CapabilityNotSupported,
    EncodedObservation,
    Observation,
    PlanBudget,
    PlanResult,
    Prediction,
    WorldModelAdapter,
)


class DummyAdapter(WorldModelAdapter):
    def encode(self, observation: Observation) -> EncodedObservation:
        self.require(Capability.ENCODE)
        return EncodedObservation(latent=observation.payload, representation_size=1)

    def update_belief(self, previous, observation, previous_action) -> BeliefState:
        self.require(Capability.BELIEF_UPDATE)
        return BeliefState(state=observation.payload, step=0 if previous is None else previous.step + 1)

    def predict(self, belief: BeliefState, actions: ActionSequence) -> Prediction:
        self.require(Capability.PREDICT)
        return Prediction(future=(belief.state, tuple(actions.actions)), horizon=len(actions.actions))

    def plan(self, belief: BeliefState, goal, budget: PlanBudget) -> PlanResult:
        self.require(Capability.PLAN)
        return PlanResult(
            actions=ActionSequence([goal]),
            objective=0.0,
            model_evaluations=1,
            latency_s=0.0,
        )


def metadata(*capabilities: Capability, model_hash: str | None = None) -> AdapterMetadata:
    return AdapterMetadata(
        name="dummy",
        version="0",
        implementation_source="internal-test",
        source_commit="test-commit",
        model_hash_sha256=model_hash,
        license_id="MIT",
        capabilities=frozenset(capabilities),
        third_party=False,
    )


class ContractTests(unittest.TestCase):
    def test_minimal_round_trip(self) -> None:
        adapter = DummyAdapter(
            metadata(
                Capability.ENCODE,
                Capability.BELIEF_UPDATE,
                Capability.PREDICT,
                Capability.PLAN,
            )
        )
        observation = Observation(payload=[1.0], modalities=("state",))
        encoded = adapter.encode(observation)
        belief = adapter.update_belief(None, observation, None)
        prediction = adapter.predict(belief, ActionSequence([0.1, 0.2]))
        plan = adapter.plan(belief, goal=0.3, budget=PlanBudget(2, 10))
        self.assertEqual(encoded.representation_size, 1)
        self.assertEqual(prediction.horizon, 2)
        self.assertEqual(plan.model_evaluations, 1)

    def test_undeclared_capability_fails_loudly(self) -> None:
        adapter = DummyAdapter(
            metadata(
                Capability.ENCODE,
                Capability.BELIEF_UPDATE,
                Capability.PREDICT,
                Capability.PLAN,
            )
        )
        with self.assertRaises(CapabilityNotSupported):
            adapter.estimate_uncertainty(Prediction(future=None, horizon=1))

    def test_invalid_hash_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DummyAdapter(metadata(Capability.ENCODE, model_hash="not-a-sha256"))

    def test_empty_action_sequence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ActionSequence([])


if __name__ == "__main__":
    unittest.main()
