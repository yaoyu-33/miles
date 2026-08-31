from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-b-cpu", labels=[])

from argparse import Namespace
from types import SimpleNamespace

import pytest

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub import nemo_gym
from miles.rollout.session.core import expose_token_id_information
from miles.utils.types import Sample


def _trajectory():
    return {
        "input_ids": [10, 11, 12, 13, 14],
        "loss_mask": [0, 0, 1, 0, 1],
        "logprobs": [0.0, 0.0, -0.1, 0.0, -0.2],
        "reward": 0.75,
    }


def test_session_exposes_exact_token_ids_when_requested():
    request = {"return_token_ids": True, "return_tokens_as_token_ids": True}
    choice = {"logprobs": {"content": [{"token": "one"}, {"token": "two"}]}}
    response = {"choices": [choice]}

    expose_token_id_information(request, response, choice, [10, 11], [12, 13])

    assert response["prompt_token_ids"] == [10, 11]
    assert choice["token_ids"] == [12, 13]
    assert choice["logprobs"]["content"] == [{"token": "token_id:12"}, {"token": "token_id:13"}]


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


def test_trajectory_preserves_length_truncation_status():
    sample = nemo_gym.apply_trajectory(Sample(index=7), _trajectory(), max_response_length=3)

    assert sample.status is Sample.Status.TRUNCATED


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

    async def fake_create(args):
        return tracer

    monkeypatch.setattr(nemo_gym, "post", fake_post)
    monkeypatch.setattr(nemo_gym.OpenAIEndpointTracer, "create", fake_create)
    sample = Sample(
        prompt=[{"role": "user", "content": "solve"}],
        metadata={"verifier_metadata": {"answer": "42"}},
    )
    state = SimpleNamespace(
        args=Namespace(
            nemo_gym_url="http://gym:8000/",
            nemo_gym_router_external_host="worker.example",
        )
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
            "policy_base_url": "http://worker.example:30000/sessions/abc/v1",
        },
        "max_retries": 3,
    }
    assert tracer.closed is True
    assert output.samples.remove_sample is True
    assert output.samples.status is Sample.Status.COMPLETED
