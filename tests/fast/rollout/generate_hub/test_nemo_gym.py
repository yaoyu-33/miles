from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-b-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import nemo_gym
from miles.utils.types import Sample


def _trajectory():
    return {
        "input_ids": [10, 11, 12, 13, 14],
        "loss_mask": [0, 0, 1, 0, 1],
        "logprobs": [0.0, 0.0, -0.1, 0.0, -0.2],
        "reward": 0.75,
    }


def test_trajectory_flows_into_native_train_data():
    sample = nemo_gym.apply_trajectory(Sample(index=7), _trajectory())
    args = SimpleNamespace(
        advantage_estimator="grpo",
        n_samples_per_prompt=1,
        reward_key=None,
        rewards_normalization=False,
        rollout_batch_size=1,
        use_dynamic_global_batch_size=False,
    )

    train_data = convert_samples_to_train_data(
        args,
        [sample],
        metadata={},
        custom_convert_samples_to_train_data_func=None,
        custom_reward_post_process_func=None,
    )

    assert train_data["tokens"] == [[10, 11, 12, 13, 14]]
    assert train_data["response_lengths"] == [3]
    assert train_data["loss_masks"] == [[1, 0, 1]]
    assert train_data["rollout_log_probs"] == [[-0.1, 0.0, -0.2]]
    assert train_data["rewards"] == [0.75]


@pytest.mark.asyncio
async def test_generate_calls_gym_run_and_uses_returned_trajectory(monkeypatch):
    captured = {}

    class FakeTracer:
        base_url = "http://session:30000/sessions/abc"
        closed = False

        async def close(self):
            self.closed = True

    tracer = FakeTracer()

    async def fake_post(url, payload, max_retries):
        captured.update(url=url, payload=payload, max_retries=max_retries)
        return {"trajectory": _trajectory(), "mask_sample": True}

    monkeypatch.setattr(nemo_gym, "post", fake_post)
    monkeypatch.setattr(nemo_gym.OpenAIEndpointTracer, "create", lambda args: _async_value(tracer))
    sample = Sample(
        prompt=[{"role": "user", "content": "solve"}],
        metadata={"verifier_metadata": {"answer": "42"}},
    )
    state = SimpleNamespace(
        args=Namespace(nemo_gym_url="http://gym:8000/", nemo_gym_max_retries=2, nemo_gym_run_timeout=60)
    )

    output = await nemo_gym.generate(
        GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params={"temperature": 0.7, "max_new_tokens": 64},
            evaluation=False,
        )
    )

    assert captured == {
        "url": "http://gym:8000/run",
        "payload": {
            "verifier_metadata": {"answer": "42"},
            "responses_create_params": {
                "input": [{"role": "user", "content": "solve"}],
                "temperature": 0.7,
                "max_output_tokens": 64,
            },
            "policy_base_url": "http://session:30000/sessions/abc/v1",
        },
        "max_retries": 2,
    }
    assert tracer.closed is True
    assert output.samples.remove_sample is True
    assert output.samples.status is Sample.Status.COMPLETED


def test_trajectory_requires_a_trainable_token():
    with pytest.raises(ValueError, match="no trainable response token"):
        nemo_gym.apply_trajectory(
            Sample(),
            {"input_ids": [1], "loss_mask": [0], "logprobs": [0.0], "reward": 0.0},
        )


async def _async_value(value):
    return value
