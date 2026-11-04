#!/usr/bin/env python3
"""Cross-check both codec directions against the official ROS C++ plugin."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'my_demo'))
from rgbd_transport import encode_depth_png, decode_depth_png


def main():
    prefixes = [Path(p) for p in os.environ.get('AMENT_PREFIX_PATH', '').split(':') if p]
    prefix = next((p for p in prefixes if (p / 'lib/libcompressed_depth_image_transport.so').is_file()), None)
    if prefix is None:
        raise SystemExit('Source ROS and install/provide ros-humble-compressed-depth-image-transport first')
    with tempfile.TemporaryDirectory(prefix='sb_codec_reference_', dir='/tmp') as directory:
        temporary = Path(directory)
        binary = temporary / 'reference'
        command = ['g++', '-std=c++17', str(ROOT / 'my_demo/tests/compressed_depth_reference.cpp'),
                   '-o', str(binary), '-I' + str(prefix / 'include'), '-I/usr/include/opencv4']
        for package in ('sensor_msgs', 'std_msgs', 'builtin_interfaces', 'rosidl_runtime_cpp',
                        'rosidl_runtime_c', 'rosidl_typesupport_interface', 'rcutils'):
            command.append('-I/opt/ros/humble/include/' + package)
        command += ['-L' + str(prefix / 'lib'), '-lcompressed_depth_image_transport']
        subprocess.run(command, check=True)
        original = np.arange(65536, dtype=np.uint16).reshape(256, 256)
        ours, reference = temporary / 'ours.bin', temporary / 'reference.bin'
        ours.write_bytes(encode_depth_png(original)[1])
        completed = subprocess.run([str(binary), str(ours), str(reference)], check=True, capture_output=True, text=True)
        report = json.loads(completed.stdout)
        decoded = decode_depth_png(SimpleNamespace(format=report['reference_format'], data=reference.read_bytes()))
        np.testing.assert_array_equal(decoded, original)
        report.update(depth_values_checked=65536, directions=['Python encode -> ROS C++ decode', 'ROS C++ encode -> Python decode'])
        output = ROOT / 'runs/sb_vision/compression_20260919'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'standard_codec_interop.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
