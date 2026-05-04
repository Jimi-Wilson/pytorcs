FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    xvfb \
    x11-xserver-utils \
    libglib2.0-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    freeglut3-dev \
    libplib-dev \
    libopenal-dev \
    libalut-dev \
    libxi-dev \
    libxmu-dev \
    libxrender-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    libpng-dev \
    libvorbis-dev \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
      rm -rf /tmp/torcs-src; \
      if git clone --depth 1 https://github.com/fmirus/torcs-1.3.7.git /tmp/torcs-src; then \
        break; \
      fi; \
      if [ "$attempt" -eq 5 ]; then \
        echo "Failed to clone TORCS source after ${attempt} attempts"; \
        exit 1; \
      fi; \
      sleep 5; \
    done
WORKDIR /tmp/torcs-src
RUN export CFLAGS="-fPIC" CPPFLAGS="-fPIC" CXXFLAGS="-fPIC" && \
    ./configure && \
    make -j1 && \
    make install && \
    make datainstall && \
    rm -rf /tmp/torcs-src

WORKDIR /workspace
COPY docker/requirements.txt /tmp/docker-requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/docker-requirements.txt

COPY . /workspace

# Mirror install_torcs/install_torcs.sh post-build steps:
#   1. Disable img_server driver (causes player-selection crash)
#   2. Install corkscrew track from upstream (not included in fmirus repo)
#   3. Patch default race configs to start on corkscrew (Docker convenience)
#   4. Set SCR default car to car1-ow1 in scr_server.xml (matches native installer)
#   5. Replace cars/car1-ow1/car1-ow1.rgb when install_torcs/assets/livery/car1-ow1.rgb exists

RUN if [ -d "/usr/local/lib/torcs/drivers/img_server" ]; then \
      mv /usr/local/lib/torcs/drivers/img_server \
         /usr/local/lib/torcs/drivers/img_server.disabled; \
    fi

RUN set -eux; \
    for attempt in 1 2 3 4 5; do \
      rm -rf /tmp/corkscrew-src; \
      if git clone --depth 1 --filter=blob:none --no-checkout \
           -b r1-3-1 \
           https://github.com/jeremybennett/torcs.git \
           /tmp/corkscrew-src; then \
        break; \
      fi; \
      [ "$attempt" -eq 5 ] && { echo "Failed to clone corkscrew source"; exit 1; }; \
      sleep 5; \
    done; \
    cd /tmp/corkscrew-src && \
    git checkout r1-3-1 -- data/tracks/road/corkscrew && \
    mkdir -p /usr/local/share/games/torcs/tracks/road/corkscrew && \
    cp -a data/tracks/road/corkscrew/. \
          /usr/local/share/games/torcs/tracks/road/corkscrew/ && \
    rm -rf /tmp/corkscrew-src

RUN python3 - <<'PY'
import re
from pathlib import Path

cfg_dir = Path("/usr/local/share/games/torcs/config/raceman")
for cfg in sorted(cfg_dir.glob("*.xml")):
    text = cfg.read_text(encoding="utf-8", errors="ignore")
    patched, n = re.subn(
        r'(<attstr\s+name="name"\s+val=")[^"]*("\s*/>(\s*<attstr\s+name="category"\s+val=")road")',
        r'\g<1>corkscrew\2',
        text,
        count=1,
    )
    if n:
        cfg.write_text(patched, encoding="utf-8")
        print(f"[docker] patched default track → corkscrew in {cfg.name}")
PY

# SCR default car: car1-ow1 (same sed as install_torcs/install_torcs.sh set_scr_server_default_car)
RUN set -eux; \
    scr_server_xml="/usr/local/share/games/torcs/drivers/scr_server/scr_server.xml"; \
    test -f "$scr_server_xml"; \
    if grep -q '<attstr name="car name" val="car1-ow1"></attstr>' "$scr_server_xml" \
        && ! grep -q '<attstr name="car name" val="car1-trb1"></attstr>' "$scr_server_xml"; then \
      echo "[docker] SCR already uses car1-ow1"; \
    else \
      sed -E -i \
        's#(<attstr name="car name" val=")car1-trb1("></attstr>)#\1car1-ow1\2#g' \
        "$scr_server_xml"; \
    fi; \
    grep -q '<attstr name="car name" val="car1-ow1"></attstr>' "$scr_server_xml"

RUN set -eux; \
    livery_src="/workspace/install_torcs/assets/livery/car1-ow1.rgb"; \
    car_dir="/usr/local/share/games/torcs/cars/car1-ow1"; \
    if [ -f "$livery_src" ]; then \
      test -d "$car_dir"; \
      echo "[docker] Replacing stock car1-ow1.rgb with project livery"; \
      rm -f "$car_dir/car1-ow1.rgb"; \
      cp -a "$livery_src" "$car_dir/car1-ow1.rgb"; \
    else \
      echo "[docker] No custom livery at $livery_src — keeping stock car1-ow1.rgb"; \
    fi

RUN chmod +x /workspace/docker/entrypoint.sh

EXPOSE 3001/udp

ENTRYPOINT ["/workspace/docker/entrypoint.sh"]
CMD ["bash"]
