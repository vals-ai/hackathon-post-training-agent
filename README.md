# hackathon-post-training-agent

This is the agent that runs your post-training checkpoint during grading. Every scored task starts this agent with your Tinker checkpoint as the model. The agent gives your model one competitive programming problem and two tools. Your model then has up to 100 turns to write and test a C++20 solution.

This repo contains the system prompt, the tool formats and the run limits that grading uses. Use it to train your model on the same formats, or to run your checkpoint locally the way grading does.

This repo does not contain:

- The scoring problems or their hidden tests. The hackathon benchmark service holds those and grades the saved solution after the run.
- Training code. You train your checkpoint yourself with Tinker.
- The agents for the other tracks. The Claude Code, Codex and mini-SWE-agent harnesses on the portal are a different thing.

The scoring copy of this agent is baked into the Daytona snapshot `hackathon-post-training-agent-v3.2`, so a run needs no Vals platform access.

## Tools

- `cpp_executor` compiles a program and runs it on stdin the model chooses (brute forces, stress tests).
- `submission` compiles a program, saves it to `/workspace/solution.cpp` and runs it on the statement's samples in `/workspace/samples/`. Only sample verdicts come back; the hidden tests are graded after the run, outside this sandbox.

At most 50 submissions and 100 turns. The last submission that compiles is graded.

## Usage

```bash
./run.sh /tmp/problem_statement.txt <task_id> tinker://.../sampler_weights/... [reasoning_effort] [temperature] [top_p] [max_output_tokens]
```

It writes `/logs/result.json`, the ATIF trajectory at `/logs/post_training_agent/atif/trajectory.json` and `/logs/post_training_agent.log`.

## Development

```bash
make install
make lint
make typecheck
make test
uv run python scripts/ensure_snapshot.py   # build the Daytona snapshot (needs DAYTONA_API_KEY)
```
