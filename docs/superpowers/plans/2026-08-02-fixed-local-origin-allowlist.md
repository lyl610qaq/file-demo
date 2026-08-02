# Fixed Local Origin Allowlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow the configured deployment origin plus exactly `http://localhost:8000` and `http://127.0.0.1:8000` to use the WebSocket and reset endpoints without weakening the existing Origin validation.

**Architecture:** Build one immutable set of normalized origins during `create_app`, using the existing strict origin parser for the configured value and two fixed local values. Store that set on application state, then use exact membership checks in both `/ws/agent` and `/api/reset`; the browser client and environment-variable contract remain unchanged.

**Tech Stack:** Python 3.12, FastAPI/Starlette, Pydantic Settings, pytest, FastAPI `TestClient`

---

## File structure

- Modify `workspace_agent/web.py`: define the fixed local origins, construct the normalized immutable allowlist, and use it for WebSocket and reset checks.
- Modify `tests/test_web.py`: add focused acceptance, rejection, normalization, and deduplication coverage for both protected endpoints.
- Modify `README.md`: document the permanent local origins and their production security trade-off.
- Keep `.env.example` unchanged: `ALLOWED_ORIGIN` remains the one deployment-specific origin setting.

### Task 1: Build the normalized origin set and use it for WebSockets

**Files:**
- Modify: `tests/test_web.py:368-470`
- Modify: `workspace_agent/web.py:49-250`
- Modify: `workspace_agent/web.py:1610-1665`
- Modify: `workspace_agent/web.py:1853-1864`

- [ ] **Step 1: Add failing WebSocket acceptance and deduplication tests**

Add these tests near the existing WebSocket origin tests in
`tests/test_web.py`:

```python
@pytest.mark.parametrize(
    "origin",
    [
        "https://file-demo-production.up.railway.app",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
)
def test_websocket_accepts_configured_and_fixed_local_origins(
    tmp_path: Path,
    origin: str,
) -> None:
    app = create_app(
        settings_for(
            tmp_path,
            allowed_origin="https://file-demo-production.up.railway.app",
        ),
        model=FinalOnlyModel(),
    )

    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/agent",
            headers={"origin": origin},
        ) as socket:
            socket.send_json({"type": "run", "task": "Inspect files"})
            events = receive_until_terminal(socket)

    assert events[-1]["type"] == "run_completed"


def test_origin_set_deduplicates_a_configured_fixed_local_origin(
    tmp_path: Path,
) -> None:
    app = create_app(
        settings_for(
            tmp_path,
            allowed_origin="HTTP://LOCALHOST:8000",
        ),
        model=FinalOnlyModel(),
    )

    assert app.state.allowed_origins == frozenset(
        {
            ("http", "localhost", 8000),
            ("http", "127.0.0.1", 8000),
        }
    )
```

- [ ] **Step 2: Run the new tests and verify they fail for the expected reasons**

Run:

```powershell
python -m pytest tests/test_web.py::test_websocket_accepts_configured_and_fixed_local_origins tests/test_web.py::test_origin_set_deduplicates_a_configured_fixed_local_origin -v
```

Expected: the configured Railway-origin case passes, the two local WebSocket
parameter cases fail with a `1008` disconnect, and the set test fails because
`app.state.allowed_origins` does not exist.

- [ ] **Step 3: Add the fixed values and construct the immutable normalized set**

Near `_ORIGIN_ERROR` in `workspace_agent/web.py`, add:

```python
_FIXED_LOCAL_ORIGIN_VALUES = (
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)
```

In `create_app`, replace:

```python
allowed_origin = _normalize_http_origin(configured.allowed_origin)
```

with:

```python
allowed_origins = frozenset(
    _normalize_http_origin(origin)
    for origin in (
        configured.allowed_origin,
        *_FIXED_LOCAL_ORIGIN_VALUES,
    )
)
```

Replace:

```python
app.state.allowed_origin = allowed_origin
```

with:

```python
app.state.allowed_origins = allowed_origins
```

This intentionally reuses `_normalize_http_origin` so malformed configured
values keep the existing safe error message.

- [ ] **Step 4: Change the WebSocket check to exact set membership**

In `/ws/agent`, replace:

```python
if request_origin != app.state.allowed_origin:
```

with:

```python
if request_origin not in app.state.allowed_origins:
```

Do not change the parsing failure path or the `1008` close behavior.

- [ ] **Step 5: Add explicit wrong-port and lookalike rejection coverage**

Add this test beside the new acceptance test:

```python
@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost.evil:8000",
    ],
)
def test_websocket_rejects_non_allowed_local_origins(
    tmp_path: Path,
    origin: str,
) -> None:
    app = create_app(
        settings_for(
            tmp_path,
            allowed_origin="https://file-demo-production.up.railway.app",
        ),
        model=FinalOnlyModel(),
    )

    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as rejected:
            with client.websocket_connect(
                "/ws/agent",
                headers={"origin": origin},
            ):
                pass

    assert rejected.value.code == 1008
```

- [ ] **Step 6: Run the focused WebSocket origin tests**

Run:

```powershell
python -m pytest tests/test_web.py -k "websocket and origin" -v
```

Expected: all selected tests pass, including the existing missing, malformed,
cross-origin, default-port normalization, and wrong-origin tests.

- [ ] **Step 7: Commit the WebSocket allowlist change**

```powershell
git add -- workspace_agent/web.py tests/test_web.py
git commit -m "feat: allow fixed local websocket origins"
```

### Task 2: Apply the same allowlist to the reset endpoint

**Files:**
- Modify: `tests/test_web.py:1222-1258`
- Modify: `workspace_agent/web.py:1765-1785`

- [ ] **Step 1: Add a failing reset acceptance test for both local origins**

Add this test immediately before
`test_reset_rejects_non_same_origin_without_running_worker`:

```python
@pytest.mark.parametrize(
    "origin",
    [
        "https://file-demo-production.up.railway.app",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
)
def test_reset_accepts_configured_and_fixed_local_origins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
) -> None:
    from workspace_agent import web

    calls = 0

    def tracked_reset(settings: Settings) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(web, "_reset_workspace", tracked_reset)
    app = create_app(
        settings_for(
            tmp_path,
            allowed_origin="https://file-demo-production.up.railway.app",
        ),
        model=FinalOnlyModel(),
    )

    with TestClient(app) as client:
        response = post_reset(client, origin=origin)

    assert response.status_code == 200
    assert response.json() == {"status": "reset"}
    assert calls == 1
```

- [ ] **Step 2: Run the test and verify the current equality check rejects it**

Run:

```powershell
python -m pytest tests/test_web.py::test_reset_accepts_configured_and_fixed_local_origins -v
```

Expected: the configured Railway-origin case passes, while both local
parameter cases fail because the endpoint returns `403` instead of `200`.

- [ ] **Step 3: Change reset protection to use the normalized set**

In `/api/reset`, replace:

```python
if request_origin != app.state.allowed_origin:
```

with:

```python
if request_origin not in app.state.allowed_origins:
```

Do not change the existing `403 ORIGIN_REJECTED` response or move the check
after rate limiting/workspace mutation.

- [ ] **Step 4: Extend the existing rejection parameter list with local edge cases**

In `test_reset_rejects_non_same_origin_without_running_worker`, extend the
`origin` list to include:

```python
        "http://localhost:8001",
        "http://127.0.0.1:8001",
        "http://localhost.evil:8000",
```

The existing `calls == 0` assertion proves rejected requests cannot invoke the
reset worker.

- [ ] **Step 5: Run all reset Origin tests**

Run:

```powershell
python -m pytest tests/test_web.py -k "reset and origin" -v
```

Expected: all selected tests pass; both fixed local origins return `200`, and
missing, malformed, wrong-port, cross-site, and lookalike origins return `403`
without running the reset worker.

- [ ] **Step 6: Commit the reset allowlist change**

```powershell
git add -- workspace_agent/web.py tests/test_web.py
git commit -m "feat: apply local origin allowlist to reset"
```

### Task 3: Document the fixed production security boundary

**Files:**
- Modify: `README.md:28-33`

- [ ] **Step 1: Update the Origin configuration documentation**

Immediately after the paragraph describing `ALLOWED_ORIGIN`, add this Chinese
paragraph to `README.md`:

```markdown
服务端还固定允许 `http://localhost:8000` 和 `http://127.0.0.1:8000` 访问 `/ws/agent` 与 `/api/reset`；其他本地端口仍会被拒绝。这个规则也存在于公网部署中，因此只有在信任本机 8000 端口所运行页面的前提下使用。不要改成通配符、后缀匹配或关闭 Origin 校验。
```

Keep the existing deployment guidance that sets `ALLOWED_ORIGIN` to the public
HTTPS origin. Do not add an `ALLOWED_ORIGINS` variable to `.env.example`.

- [ ] **Step 2: Run documentation and distribution tests**

Run:

```powershell
python -m pytest tests/test_distribution_docs.py -v
```

Expected: all distribution documentation tests pass.

- [ ] **Step 3: Commit the documentation change**

```powershell
git add -- README.md
git commit -m "docs: explain fixed local origin access"
```

### Task 4: Verify the complete change

**Files:**
- Verify: `workspace_agent/web.py`
- Verify: `tests/test_web.py`
- Verify: `README.md`

- [ ] **Step 1: Check formatting and the final diff**

Run:

```powershell
git diff --check HEAD~3..HEAD
git status --short --branch
```

Expected: `git diff --check` exits `0`; only the pre-existing untracked
`agent.bat` may remain, and no feature files are modified or untracked.

- [ ] **Step 2: Run the focused Origin regression suite**

Run:

```powershell
python -m pytest tests/test_web.py -k "origin" -v
```

Expected: every selected Origin test passes.

- [ ] **Step 3: Run the complete test suite without loading the developer `.env`**

The local `.env` intentionally contains deployment-specific values that alter
settings tests. Temporarily move it and restore it in `finally`:

```powershell
$taskEnv = Join-Path (Get-Location) '.env'
$taskBackup = Join-Path (Get-Location) '.env.codex-test-backup'
$taskMovedEnv = $false
$taskExit = 1
try {
    if (Test-Path -LiteralPath $taskBackup) {
        throw "Temporary backup already exists: $taskBackup"
    }
    if (Test-Path -LiteralPath $taskEnv) {
        Move-Item -LiteralPath $taskEnv -Destination $taskBackup
        $taskMovedEnv = $true
    }
    python -m pytest -q
    $taskExit = $LASTEXITCODE
}
finally {
    if ($taskMovedEnv) {
        Move-Item -LiteralPath $taskBackup -Destination $taskEnv
    }
}
exit $taskExit
```

Expected: all tests pass with only the repository's documented skips.

- [ ] **Step 4: Review the exact acceptance criteria against evidence**

Confirm from the focused test output that:

```text
configured Railway origin: accepted
http://localhost:8000: accepted
http://127.0.0.1:8000: accepted
http://localhost:8001: rejected
http://127.0.0.1:8001: rejected
http://localhost.evil:8000: rejected
missing or malformed Origin: rejected
```

- [ ] **Step 5: Confirm the final repository state**

Run:

```powershell
git log -4 --oneline --decorate
git status --short --branch
```

Expected: the three implementation commits are present, feature files are
clean, and the unrelated untracked `agent.bat` remains untouched.

The implementation is then ready to push and redeploy. Pushing is a separate
explicit action and should not occur unless requested.
