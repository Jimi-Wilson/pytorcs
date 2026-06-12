#!/bin/bash

while true; do
    /torcs/BUILD/bin/torcs -r /root/.torcs/config/raceman/practise.xml -nofuel -nodamage -nolaptime -t 50000000

    echo "[$(date -u +%H:%M:%SZ)] TORCS process exited. Relaunching..."
done