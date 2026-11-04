#!/usr/bin/env python3
"""Prepare a common-source timesyncd client configuration; apply only with --apply.

Default invocation only writes reviewable files in the project. --apply requires
root and is intended for the later multi-host deployment, not the local NTP lab.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock_sync import ClockSettings, validate_server
ROOT = Path(__file__).resolve().parents[2]


def render_timesyncd(server):
    server = validate_server(server)
    return ('# FoundationPose: use the same NTP source on robot, vision and control hosts.\n'
            '[Time]\nNTP=\nNTP=' + server + '\nFallbackNTP=\n'
            'PollIntervalMinSec=16\nPollIntervalMaxSec=64\nConnectionRetrySec=15\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server', required=True, help='common production NTP server IP/hostname')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'runs/sb_vision/clock_sync_setup')
    parser.add_argument('--apply', action='store_true', help='install the reviewed drop-in and restart timesyncd (root required)')
    args = parser.parse_args()
    server = validate_server(args.server)
    content = render_timesyncd(server)
    settings = asdict(ClockSettings(server=server))
    if not args.apply:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'90-sb-vision.conf').write_text(content)
        (args.output_dir/'clock_sync.json').write_text(json.dumps(settings, indent=2)+'\n')
        print(json.dumps({'applied':False,'server':server,'preview_directory':str(args.output_dir.resolve()),
                          'note':'No service, system clock or project default configuration was changed.'},indent=2))
        return 0
    if os.geteuid() != 0:
        raise SystemExit('--apply requires root; review the generated files, then run this command with sudo in your terminal')
    if subprocess.run(['systemctl','is-active','--quiet','chrony']).returncode == 0:
        raise SystemExit('chrony is active: configure its selected server instead of starting a competing timesyncd service')
    target = Path('/etc/systemd/timesyncd.conf.d/90-sb-vision.conf')
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    existed = target.exists()
    if existed:
        backup = target.with_name(target.name+'.bak-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        shutil.copy2(target, backup)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', dir=target.parent, prefix='.sb-timesync-', delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
        temporary.chmod(0o644)
        os.replace(temporary, target)
        subprocess.run(['systemctl','enable','systemd-timesyncd'],check=True)
        subprocess.run(['systemctl','restart','systemd-timesyncd'],check=True)
    except BaseException:
        if backup:
            shutil.copy2(backup, target)
        elif not existed:
            target.unlink(missing_ok=True)
        subprocess.run(['systemctl','restart','systemd-timesyncd'],check=False)
        raise
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    print(json.dumps({'applied':True,'server':server,'drop_in':str(target),'backup':str(backup) if backup else None,
                      'note':'Service configured; run check_clock_sync.py to measure readiness before publishing poses.'},indent=2))
    return 0


if __name__ == '__main__': raise SystemExit(main())
