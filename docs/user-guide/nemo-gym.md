---
title: NeMo Gym
description: Train on NVIDIA NeMo Gym environments through its token-aligned trajectory API.
---

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) is NVIDIA's RL environment
ecosystem: environments are HTTP *resources servers* (code execution, search,
SWE tasks, ...) paired with *agents* that drive an episode end-to-end and
grade it. Task containers run through NeMo Gym's own sandbox provider API
(`nemo_gym.sandbox`) — Docker locally, or Daytona / Apptainer / ECS Fargate /
OpenSandbox — selected by config, no agent changes.

Miles POSTs each sample to a NeMo Gym agent server's `/run` endpoint with
`policy_base_url` set to a session's OpenAI-compatible URL. Gym returns the
finished rollout as four token-aligned fields: `input_ids`, `loss_mask`,
`logprobs`, and `reward`. The connector maps those fields directly onto a
Miles `Sample`; Miles' normal sample-to-training conversion handles the rest.

## Try it

The maintained recipe is **SWE-bench GRPO with mini-swe-agent** in
[`examples/experimental/nemo-gym`](https://github.com/radixark/miles/tree/main/examples/experimental/nemo-gym).
In short:

1. **Environment side** — on any docker-capable host, clone NeMo Gym `main`
   (>= `fcca3a8`) and start the `mini_swe_agent_2` responses-API agent server
   with the docker sandbox provider config.
2. **Data** — convert SWE-bench Verified to Miles prompt data with
   `download_and_process_data.py`; the task instance rides in each sample's
   `metadata`.
3. **Training side** — point `NEMO_GYM_URL` at the agent server and launch
   `run.py`, wiring the connector:

```bash
--custom-generate-function-path miles.rollout.generate_hub.nemo_gym.generate
--nemo-gym-url http://<gym-host>:12000
--use-session-server
--tito-model qwen3
```

The connector test exercises Gym's `/run` boundary and Miles' real
sample-to-training conversion. The earlier session-reconstruction recipe was
validated in a 4-GPU training smoke; the simplified trajectory-return path
still needs a fresh live GPU smoke. Follow the
[recipe README](https://github.com/radixark/miles/blob/main/examples/experimental/nemo-gym/README.md)
for the NeMo Gym server setup and launch walkthrough.
