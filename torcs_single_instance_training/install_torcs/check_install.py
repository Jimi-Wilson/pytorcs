#!/usr/bin/env python3
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SEARCH_BASES = [
    Path("/usr/local/share/games/torcs"),
    Path("/usr/share/games/torcs"),
    Path.home() / ".torcs",
]


def check_torcs_binary():
    torcs_path = shutil.which("torcs")
    if not torcs_path:
        return False, "torcs binary not found in PATH."
    return True, f"torcs binary found at {torcs_path}"


def has_scr_markers(base: Path) -> bool:
    if not base.exists():
        return False
    targets = [base / "drivers", base / "config" / "drivers", base]
    for root in targets:
        if not root.exists():
            continue
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            name = file_path.name.lower()
            if "scr" in name:
                return True
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "scr" in text.lower() or "3101" in text:
                return True
    return False


def check_scr_capability():
    for base in SEARCH_BASES:
        if has_scr_markers(base):
            return True, f"SCR markers found in {base}"
    try:
        result = subprocess.run(
            ["dpkg", "-L", "torcs"], capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        result = None
    if result and result.returncode == 0 and "scr" in result.stdout.lower():
        return True, "SCR markers found in package file list"
    return False, "No SCR markers found in standard TORCS data directories."


def check_corkscrew():
    for base in SEARCH_BASES:
        track_dir = base / "tracks" / "road" / "corkscrew"
        if (track_dir / "corkscrew.xml").exists():
            return True, f"Corkscrew track found at {track_dir}"
    return False, "Corkscrew track not found in installed TORCS data directories."


def check_scr_default_car():
    for base in SEARCH_BASES:
        scr_server_xml = base / "drivers" / "scr_server" / "scr_server.xml"
        if not scr_server_xml.exists():
            continue
        try:
            text = scr_server_xml.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return False, f"Could not read {scr_server_xml}: {exc}"

        has_target = '<attstr name="car name" val="car1-ow1"></attstr>' in text
        has_old = '<attstr name="car name" val="car1-trb1"></attstr>' in text
        if has_target and not has_old:
            return True, f"SCR default car is car1-ow1 in {scr_server_xml}"
        if has_target and has_old:
            return (
                False,
                f"Mixed SCR car mapping in {scr_server_xml} (contains both car1-ow1 and car1-trb1)",
            )
        return False, f"SCR default car is not car1-ow1 in {scr_server_xml}"

    return False, "scr_server.xml not found in standard TORCS data directories."


def main() -> int:
    checks = [
        ("torcs-path", check_torcs_binary),
        ("scr-capable", check_scr_capability),
        ("scr-default-car", check_scr_default_car),
        ("corkscrew-track", check_corkscrew),
    ]
    all_ok = True
    for name, fn in checks:
        ok, msg = fn()
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        all_ok = all_ok and ok
    if all_ok:
        print("All checks passed.")
        return 0
    print("One or more checks failed. Re-run: bash install_torcs/install_torcs.sh")
    return 1


if __name__ == "__main__":
    sys.exit(main())
