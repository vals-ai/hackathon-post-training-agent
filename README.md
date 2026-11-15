# hackathon-post-training-agent

Valkyrie agent for the post-training track. It solves one competitive programming problem with a Tinker checkpoint, compiling and testing C++20 inside its own sandbox. The harness is baked into the Daytona snapshot `hackathon-post-training-agent-v3.2`, so a run needs no Vals platform access; graded by the hackathon benchmark service.

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
