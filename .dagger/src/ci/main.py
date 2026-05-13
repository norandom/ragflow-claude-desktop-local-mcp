"""Dagger CI module for ragflow-claude-mcp.

Functions:
    test  — run pytest in a clean container
    build — produce wheel + sdist, return dist/ Directory
    ci    — test then build (used by the release workflow)

Invoke from a checkout root:
    dagger call test  --source=.
    dagger call build --source=. export --path=./dist
    dagger call ci    --source=. export --path=./dist
"""

import dagger
from dagger import dag, function, object_type

PYTHON_IMAGE = "python:3.12-slim"


def _base(source: dagger.Directory) -> dagger.Container:
    """Container with uv installed, project mounted at /src, cwd=/src.

    Cache mounts speed up repeat invocations:
      - uv cache (~/.cache/uv) — package resolver
      - uv tool cache (~/.local/share/uv) — installed-tool state
    """
    return (
        dag.container()
        .from_(PYTHON_IMAGE)
        .with_exec(["pip", "install", "--quiet", "--no-cache-dir", "uv"])
        .with_mounted_cache("/root/.cache/uv", dag.cache_volume("uv-cache"))
        .with_mounted_directory("/src", source)
        .with_workdir("/src")
    )


@object_type
class Ci:
    @function
    async def test(self, source: dagger.Directory) -> str:
        """Run the project's pytest suite in a clean container. Returns pytest stdout."""
        return await (
            _base(source)
            .with_exec(["uv", "sync", "--extra", "dev"])
            .with_exec(["uv", "run", "pytest", "tests/", "-q", "--no-header"])
            .stdout()
        )

    @function
    def build(self, source: dagger.Directory) -> dagger.Directory:
        """Build wheel + sdist with uv. Returns dist/ directory."""
        return (
            _base(source)
            .with_exec(["uv", "build"])
            .directory("/src/dist")
        )

    @function
    async def ci(self, source: dagger.Directory) -> dagger.Directory:
        """Test then build. Fails fast if tests fail. Returns dist/."""
        await self.test(source)
        return self.build(source)
