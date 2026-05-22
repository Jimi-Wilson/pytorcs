# Docker Setup (TORCS + SCR): All OS

This guide gives GUI-first, working steps to run TORCS + SCR + Python tooling in Docker on Linux, Windows, and macOS.

## What this setup provides

- Ubuntu 22.04 container
- TORCS 1.3.7 + SCR built from `fmirus/torcs-1.3.7`
- Python dependencies from `docker/requirements.txt`
- SCR UDP port mapping on `3001`
- GUI mode support (required)
- Docker audio disabled by design in TORCS (not an error)
- TORCS is launched via [libstrangle](https://github.com/milaq/libstrangle) at **60 FPS** by default (`TORCS_FPS_LIMIT`, set to `0` to disable). Xvfb does not provide vsync, so this caps an otherwise unbounded render loop and keeps SCR timing steadier on fast CPUs.

## Common steps (all operating systems)

From repository root:

```bash
docker compose -f docker/docker-compose.yml build
```

## Linux (Ubuntu and similar): full GUI steps

1. Allow local Docker containers to access your X server:

```bash
xhost +local:docker
xhost +SI:localuser:root
```

2. Start TORCS with host display forwarding:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e DISPLAY="${DISPLAY:-:0}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  torcs-dev torcs -t 100000
```

3. After finishing (recommended), revoke permission:

```bash
xhost -local:docker
xhost -SI:localuser:root
```

## Windows (WSL2 + Docker Desktop): full GUI steps

1. Install and start an X server on Windows (VcXsrv recommended).
2. In WSL2 shell, set `DISPLAY` to the Windows host IP:

```bash
export DISPLAY="$(awk '/nameserver/{print $2; exit}' /etc/resolv.conf):0.0"
```

3. Start TORCS:

```bash
docker compose -f docker/docker-compose.yml run --rm torcs-dev torcs -t 100000
```

Notes:
- Keep VcXsrv running while TORCS is running.
- If firewall prompts appear, allow local/private network access.
- If your shell has a stale value like `DISPLAY=:0`, overwrite it with the `nameserver` command above before running Docker.
- In VcXsrv, start with `Disable access control` enabled for initial setup (tighten later if needed).

## macOS (Docker Desktop + XQuartz): full GUI steps

1. Install XQuartz and start it once.
2. In XQuartz preferences, enable:
   - `Allow connections from network clients`
3. Restart XQuartz after changing that setting.
4. Allow Docker containers to connect to XQuartz (disables access control for local dev):

```bash
xhost +
```

5. Start TORCS with XQuartz forwarding:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e DISPLAY=host.docker.internal:0 \
  torcs-dev torcs -t 100000
```

## Start a development shell in container

```bash
docker compose -f docker/docker-compose.yml run --rm torcs-dev bash
```

Inside the container:

```bash
python3 -c "import gymnasium, numpy, dotenv, stable_baselines3; print('python deps ok')"
```

Run TORCS manually:

```bash
torcs -t 100000
```

## SCR port usage

- Default SCR port is `3001`
- Compose maps `3001:3001/udp`
- If your host already uses this port, change host mapping in `docker/docker-compose.yml` (for example `3002:3001/udp`)

## Troubleshooting

- **`freeglut ... failed to open display`**
  - Linux: ensure `/tmp/.X11-unix` is mounted and run both `xhost +local:docker` and `xhost +SI:localuser:root`.
  - Windows/WSL2: this is usually the same root cause as Linux (container cannot access host display). Use this exact sequence:
    - `export DISPLAY="$(awk '/nameserver/{print $2; exit}' /etc/resolv.conf):0.0"`
    - Start VcXsrv with `Disable access control`.
    - Re-run `docker compose -f docker/docker-compose.yml run --rm torcs-dev torcs -t 100000`
  - Windows/WSL2: if still failing, check `echo $DISPLAY` (must not be `:0`), and allow private network access in Windows Firewall for VcXsrv.
  - macOS: verify XQuartz is running, network clients are allowed, and `DISPLAY=host.docker.internal:0`.

- **No TORCS sound in Docker**
  - Expected behavior. Audio is intentionally disabled in Docker TORCS.
  - This is not a runtime failure.

- **`docker compose` command not found**
  - Update Docker Desktop / Docker Engine and use Compose V2 (`docker compose`).

- **Port conflict on `3001`**
  - Change host-side mapping in `docker/docker-compose.yml`.

- **Container exits immediately**
  - Run interactive shell for inspection:
    - `docker compose -f docker/docker-compose.yml run --rm torcs-dev bash`

## Notes

- GUI mode is required for this workflow.
- This dev compose is intentionally single-container and separate from future multi-simulator training compose files.
