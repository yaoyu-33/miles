# SWE-agent training via NeMo Gym

This demo trains a SWE agent with Miles while NeMo Gym owns the sandbox,
agent loop, and verification. Gym's `/run` response now includes the exact
token-aligned training trajectory, so Miles no longer reconstructs it from a
second session-server read or carries reward through a separate hook.

```text
Miles Sample
  -> NeMo Gym /run
       -> {input_ids, loss_mask, logprobs, reward}
  -> Miles Sample
  -> existing Sample -> train_data conversion
```

The only connector is
`miles.rollout.generate_hub.nemo_gym.generate`. It creates a Miles session URL
for the external agent's policy calls, sends that URL to Gym, and maps Gym's
four returned fields onto Miles' existing `Sample` fields:

| NeMo Gym | Miles |
| --- | --- |
| `input_ids` | `tokens` |
| response suffix of `loss_mask` | `loss_mask` |
| response suffix of `logprobs` | `rollout_log_probs` |
| `reward` | `reward` |

The session server remains the policy endpoint used by the external SWE
agent. It is no longer also a second trajectory assembly path.

## Start NeMo Gym

From a NeMo Gym checkout, start `mini_swe_agent_2` with a sandbox provider:

```bash
gym env start \
  --config responses_api_agents/mini_swe_agent_2/configs/mini_swe_agent_2.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --model-type vllm_model \
  '++mini_swe_agent_2.responses_api_agents.mini_swe_agent_2.port=12000'
```

## Prepare data

```bash
python examples/experimental/nemo-gym/download_and_process_data.py \
  --input princeton-nlp/SWE-bench_Verified \
  --split test \
  --subset verified \
  --output /root/swe_verified.jsonl
```

Each output row has a `prompt` and the complete SWE instance in `metadata`.
The connector forwards that metadata at the top level of Gym's `/run`
request and replaces only `responses_create_params.input` with the prompt.

## Run

```bash
export NEMO_GYM_URL=http://<gym-host>:12000

# Set this only when Gym cannot resolve the trainer hostname.
export MILES_ROUTER_EXTERNAL_HOST=<trainer-host-reachable-from-gym>

python examples/experimental/nemo-gym/run.py
```

The essential Miles flags are now:

```bash
--custom-generate-function-path miles.rollout.generate_hub.nemo_gym.generate
--nemo-gym-url http://<gym-host>:12000
--use-session-server
--session-server-ip 0.0.0.0
--tito-model qwen3
```

There is no NeMo Gym-specific agent function, reward function, or trajectory
postprocessor. Transport failures mark the sample aborted; the demo's
`check_no_aborted` filter drops the affected prompt group.

## Validate the connector

```bash
pytest -q tests/fast/rollout/generate_hub/test_nemo_gym.py
```

The test calls the Gym boundary and then runs Miles' real
`convert_samples_to_train_data`, asserting the final tokens, response length,
loss mask, rollout logprobs, and reward.
