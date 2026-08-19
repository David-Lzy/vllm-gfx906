# Release process

## Branches

- `main`: hardware-validated gfx906 releases and documentation
- `integration/vX.Y.Z`: exact upstream release used for integration
- `port/<feature>-vX.Y`: one isolated compatibility or optimization port
- `agent/<topic>`: short-lived documentation or maintenance branches

Upstream versions are fetched explicitly. They do not advance `main`
automatically.

## Versioning

The first target image is:

```text
ghcr.io/david-lzy/vllm-gfx906:v0.26.0-gfx906.1
```

Patch releases increment the final gfx906 revision. Images must also record
the source commit and immutable image digest.

## Validation flow

1. Rebase or recreate the integration branch at an exact upstream tag.
2. Port one feature per branch and pull request.
3. Run lint and GPU-free tests in hosted CI.
4. Build and run MI50 tests only from trusted branches on the private runner.
5. Execute the full benchmark protocol and publish the evidence summary.
6. Merge into `main` only after hardware review.
7. Publish the versioned GHCR image.
8. Start a temporary-port canary and run a 30-60 minute soak.
9. Switch production only after health, text, image, metrics, and logs pass.

Public pull requests must never trigger untrusted code on the private MI50
runner.

## Rollback

Before a production switch, preserve:

- The prior image digest
- The prior deployment configuration
- The production model cache
- A tested command to restore the prior service

Rollback is required for repeated HTTP 500 responses, OOM, fatal RCCL errors,
stalled work, quality regression, or core throughput below the accepted gate.
The new image remains a candidate until the canary and production soak finish.
