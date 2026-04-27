# Windows Docker Setup (TORCS + SCR)

This guide lets you run TORCS + SCR + Python tooling on Windows without a native Linux install.

## Prerequisites

- Docker Desktop installed
- Docker Desktop configured to use WSL2 backend
- Git for Windows (or WSL2 git) to clone the repository

Optional for GUI mode:
- VcXsrv (or another X server on Windows)

## What this setup provides

- Ubuntu 22.04 container
- TORCS 1.3.7 + SCR built from `fmirus/torcs-1.3.7`
- Python dependencies from `docker/requirements.txt`
- Headless display support via Xvfb (works out of the box)
- SCR UDP port mapping on `3001`

## Quick start (headless default)

From repository root:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run --rm torcs-dev torcs -t 100000
```

This should start TORCS in text/headless mode inside the container.

## Start a development shell in container

```bash
docker compose -f docker/docker-compose.yml run --rm torcs-dev bash
```

Inside the container:

```bash
python3 -c "import gymnasium, numpy, dotenv, stable_baselines3; print('python deps ok')"
```

## SCR port usage

- Default SCR port is `3001`
- Compose maps `3001:3001/udp`
- If your host already uses this port, stop the conflicting process or change the compose port mapping

## Optional GUI mode on Windows

GUI forwarding is optional. Headless mode is supported by default via Xvfb.

To attempt GUI mode:

1. Start VcXsrv on Windows (disable access control for quick local testing).
2. Set `DISPLAY` so the container can reach your X server:
   - In WSL2, this is often your Windows host IP (for example `DISPLAY=<windows-host-ip>:0`).
3. Run container with that `DISPLAY` exported in your shell before `docker compose`.

If `DISPLAY` is not set, entrypoint starts Xvfb on `:1` automatically.

## Troubleshooting

- **Build fails during apt install**
  - Ensure internet access from Docker and rerun build.

- **`docker compose` command not found**
  - Update Docker Desktop; use integrated Compose V2 (`docker compose`, not `docker-compose`).

- **Port conflict on 3001**
  - Change host-side mapping in `docker/docker-compose.yml`, e.g. `3002:3001/udp`.

- **GUI does not open**
  - Verify VcXsrv is running and firewall permits it.
  - Verify `DISPLAY` is set correctly.
  - Fall back to headless (`torcs -t 100000`) to confirm core setup works.

- **Container exits immediately**
  - Run with explicit command:
    - `docker compose -f docker/docker-compose.yml run --rm torcs-dev bash`

## Notes

- This dev compose is intentionally single-container and separate from future multi-simulator training compose files.
