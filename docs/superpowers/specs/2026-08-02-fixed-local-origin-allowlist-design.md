# Fixed Local Origin Allowlist Design

## Context

The application currently accepts exactly one configured HTTP origin through
`ALLOWED_ORIGIN`. Both the `/ws/agent` WebSocket handshake and the
`/api/reset` mutation endpoint compare the request `Origin` against that one
normalized value. A Railway deployment therefore accepts its public origin but
rejects local browser pages served from `localhost:8000` or
`127.0.0.1:8000`.

The selected approach keeps the existing deployment-specific
`ALLOWED_ORIGIN` setting and adds two fixed local development origins in code.

## Goals

- Accept the configured deployment origin.
- Also accept exactly `http://localhost:8000` and
  `http://127.0.0.1:8000`.
- Apply the same origin policy to `/ws/agent` and `/api/reset`.
- Preserve strict rejection of missing, malformed, cross-site, and
  wrong-port origins.
- Preserve existing origin normalization and the `ALLOWED_ORIGIN`
  configuration contract.

## Non-goals

- Supporting arbitrary localhost ports.
- Adding wildcard, suffix, regex, or reflected-origin matching.
- Adding a new `ALLOWED_ORIGINS` environment variable.
- Changing authentication, rate limiting, proxy trust, or CORS behavior.

## Design

Define an immutable pair of fixed local origin strings in
`workspace_agent.web`:

```text
http://localhost:8000
http://127.0.0.1:8000
```

During `create_app`, normalize the configured `ALLOWED_ORIGIN` and both fixed
local values with the existing `_normalize_http_origin` function. Store the
results as an immutable origin set on application state. Set semantics remove
duplicates when the configured origin is already one of the fixed local
origins.

Replace the current equality checks in `/ws/agent` and `/api/reset` with exact
membership checks against that normalized set. Parsing failures continue to be
treated as rejected origins. WebSocket failures continue to close with code
`1008`; HTTP reset failures continue to return `403 ORIGIN_REJECTED`.

The browser client remains unchanged because it already constructs the
WebSocket URL from the page's current host.

## Security boundary

Origin validation remains enabled. The implementation will not accept
wildcards, arbitrary ports, missing headers, malformed values, lookalike
domains, or `ws://` values in the HTTP `Origin` header.

The accepted trade-off is that every deployment, including production, trusts
pages served from the user's local machine on port 8000. A malicious or
compromised local service on that exact port could therefore connect to the
deployment. This is the explicit consequence of choosing fixed local origins
instead of a deployment-configured allowlist.

## Testing

Automated tests will verify that:

- WebSocket connections succeed from the configured origin.
- WebSocket connections succeed from both fixed local origins.
- `/api/reset` accepts both fixed local origins.
- Origins using another localhost port are rejected.
- Missing and malformed origins remain rejected.
- Lookalike remote hosts remain rejected.
- Origin normalization still handles scheme/host case and default ports.
- A configured origin that duplicates a fixed local origin causes no error.

The focused WebSocket and reset-origin tests will run first, followed by the
complete test suite.

## Documentation and deployment

Update `README.md` to document the two permanently allowed local origins and
their security trade-off. Keep `.env.example` and Railway configuration on the
single `ALLOWED_ORIGIN` variable; Railway should continue to set it to the
public HTTPS origin.

No Railway networking, port, or volume changes are required for this feature.
After the code is pushed, Railway must deploy the new commit.

## Acceptance criteria

- A page served from `http://localhost:8000` can establish `/ws/agent`.
- A page served from `http://127.0.0.1:8000` can establish `/ws/agent`.
- The configured Railway public origin continues to work.
- An otherwise identical local origin on a non-8000 port is rejected.
- Existing origin error behavior and all automated tests remain passing.
