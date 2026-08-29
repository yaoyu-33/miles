"""Use a NeMo Gym agent as a Miles custom generation function."""

import argparse
import asyncio
import logging
import os
from copy import deepcopy
from typing import Any
from urllib.parse import urlparse, urlunparse

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.openai_endpoint_utils import OpenAIEndpointTracer
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def apply_trajectory(sample: Sample, trajectory: dict[str, Any]) -> Sample:
    """Map Gym's token-aligned trajectory onto Miles' native sample."""
    input_ids = list(trajectory["input_ids"])
    loss_mask = list(trajectory["loss_mask"])
    logprobs = list(trajectory["logprobs"])
    if not input_ids or len(input_ids) != len(loss_mask) or len(input_ids) != len(logprobs):
        raise ValueError("NeMo Gym trajectory fields must be non-empty and token-aligned")
    try:
        response_start = loss_mask.index(1)
    except ValueError as error:
        raise ValueError("NeMo Gym trajectory has no trainable response token") from error

    sample.tokens = input_ids
    sample.response_length = len(input_ids) - response_start
    sample.loss_mask = loss_mask[response_start:]
    sample.rollout_log_probs = logprobs[response_start:]
    sample.reward = float(trajectory["reward"])
    sample.status = Sample.Status.COMPLETED
    sample.validate()
    return sample


def _run_request(sample: Sample, sampling_params: dict[str, Any]) -> dict[str, Any]:
    request = deepcopy(sample.metadata.get("nemo_gym_run_request", sample.metadata))
    responses_create_params = request.setdefault("responses_create_params", {})
    responses_create_params["input"] = sample.prompt
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("max_new_tokens", "max_output_tokens"),
    ):
        if source in sampling_params:
            responses_create_params.setdefault(target, sampling_params[source])
    return request


def _policy_base_url(session_base_url: str) -> str:
    policy_base_url = f"{session_base_url}/v1"
    if external_host := os.getenv("MILES_ROUTER_EXTERNAL_HOST"):
        parsed = urlparse(policy_base_url)
        netloc = f"{external_host}:{parsed.port}" if parsed.port else external_host
        policy_base_url = urlunparse(parsed._replace(netloc=netloc))
    return policy_base_url


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    """Call Gym's wired ``/run`` endpoint and return a native Miles sample."""
    sample = input.sample
    if sample.multimodal_inputs:
        raise ValueError("The minimal NeMo Gym trajectory connector is text-only")

    tracer = await OpenAIEndpointTracer.create(input.args)
    request = _run_request(sample, input.sampling_params)
    request["policy_base_url"] = _policy_base_url(tracer.base_url)
    try:
        result = await asyncio.wait_for(
            post(
                f"{input.args.nemo_gym_url.rstrip('/')}/run",
                request,
                max_retries=input.args.nemo_gym_max_retries,
            ),
            timeout=input.args.nemo_gym_run_timeout,
        )
    except Exception as error:
        logger.warning("NeMo Gym /run failed: %s", error)
        sample.status = Sample.Status.ABORTED
        return GenerateFnOutput(samples=sample)
    finally:
        await tracer.close()

    trajectory = result.get("trajectory")
    if trajectory is None:
        raise ValueError("NeMo Gym /run response did not include trajectory")
    sample.remove_sample = bool(result.get("mask_sample", False))
    sample.metadata["eval_report"] = result.get("metadata", {}) or {}
    return GenerateFnOutput(samples=apply_trajectory(sample, trajectory))


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nemo-gym-url", required=True)
    parser.add_argument("--nemo-gym-max-retries", type=int, default=3)
    parser.add_argument("--nemo-gym-run-timeout", type=float, default=3600)


generate.add_arguments = _add_arguments
