import asyncio
from pathlib import Path

from daytona import AsyncDaytona, CreateSnapshotParams, Image, Resources

NAME = "hackathon-post-training-agent-v3.2"
DOCKERFILE = Path(__file__).resolve().parents[1] / "Dockerfile"


async def main():
    async with AsyncDaytona() as daytona:
        page = 1
        while True:
            result = await daytona.snapshot.list(page=page, limit=100)
            hit = [s for s in result.items if s.name == NAME]
            if hit or page >= result.total_pages:
                break
            page += 1
        snapshot = (
            hit[0]
            if hit
            else await daytona.snapshot.create(
                CreateSnapshotParams(
                    name=NAME,
                    image=Image.from_dockerfile(DOCKERFILE).entrypoint(["sleep", "infinity"]),
                    resources=Resources(cpu=4, memory=4, disk=10),
                ),
                on_logs=lambda chunk: print(chunk, end="", flush=True),
                timeout=0,
            )
        )
        while str(snapshot.state).split(".")[-1].lower() not in {"active", "error", "build_failed"}:
            print("state", snapshot.state, flush=True)
            await asyncio.sleep(20)
            snapshot = await daytona.snapshot.get(NAME)
        print("FINAL", NAME, snapshot.state, getattr(snapshot, "error_reason", None))


asyncio.run(main())
