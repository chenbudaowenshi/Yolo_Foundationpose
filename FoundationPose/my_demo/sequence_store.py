"""Crash-safe uint32 publication sequences for one ROS domain/topic stream.

A committed high-water mark reserves a range before any value is returned.
A restart abandons the unused part of that range. The lock inode is kept stable
while the JSON record is replaced atomically. Storage must be a persistent local
filesystem supporting flock, atomic rename and fsync (including directories).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading

MAX_SEQUENCE = 2**32 - 1


class SequenceStateError(RuntimeError):
    pass


def uint32(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SEQUENCE:
        raise ValueError(f'{name} must be an integer in [0, {MAX_SEQUENCE}]')
    return value


def _canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path):
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
        _sync_directory(directory.parent)


class PersistentSequence:
    def __init__(self, directory, *, domain_id, topic, start_after=None, block_size=4096):
        if isinstance(domain_id, bool) or not isinstance(domain_id, int) or domain_id < 0:
            raise ValueError('domain_id must be a nonnegative integer')
        if not isinstance(topic, str) or not topic.startswith('/'):
            raise ValueError('topic must be the resolved absolute ROS topic name')
        if isinstance(block_size, bool) or not isinstance(block_size, int) or not 1 <= block_size <= MAX_SEQUENCE:
            raise ValueError('block_size must be a positive uint32-sized integer')
        if start_after is not None:
            uint32(start_after, 'start_after')
        self.directory = Path(directory).expanduser().resolve()
        self.stream = {'domain_id': domain_id, 'topic': topic}
        name = f'domain_{domain_id}_{hashlib.sha256(topic.encode()).hexdigest()[:24]}'
        self.path = self.directory / (name + '.json')
        self.lock_path = self.directory / (name + '.lock')
        self.block_size = block_size
        self._mutex = threading.Lock()
        self._fd = None
        self._closed = False
        self._failed = False
        self._current = self._reserved = 0
        try:
            _mkdir_durable(self.directory)
            flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | getattr(os, 'O_NOFOLLOW', 0)
            self._fd = os.open(self.lock_path, flags, 0o600)
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SequenceStateError(f'Another publisher owns sequence stream {self.stream}: {self.lock_path}') from exc
            marker = os.pread(self._fd, 4096, 0)
            expected_marker = _canonical({'version': 1, 'initialized': True, 'stream': self.stream})
            if marker and marker != expected_marker:
                raise SequenceStateError(f'Invalid sequence lock marker; refusing to reset: {self.lock_path}')
            if self.path.is_symlink():
                raise SequenceStateError(f'Sequence state must not be a symlink: {self.path}')
            if self.path.exists():
                high = self._read_state()
                if start_after is not None:
                    high = max(high, start_after)  # bootstrap flag can never lower durable history
            else:
                if marker:
                    raise SequenceStateError(f'Sequence state disappeared; refusing to restart at zero: {self.path}')
                if start_after is None:
                    raise SequenceStateError('Sequence stream is not initialized. On first deployment use '
                        '--sequence-start-after N (N=0 for a fresh receiver; otherwise its last accepted sequence). '
                        f'State: {self.path}')
                high = start_after
            self._current = self._reserved = high
            self._reserve()  # durability barrier before this publisher can emit even its first message
            if not marker:
                os.lseek(self._fd, 0, os.SEEK_SET)
                view = memoryview(expected_marker)
                while view:
                    count = os.write(self._fd, view)
                    if count <= 0:
                        raise OSError('unable to write sequence initialization marker')
                    view = view[count:]
                os.ftruncate(self._fd, len(expected_marker))
                os.fsync(self._fd)
                _sync_directory(self.directory)
        except BaseException:
            self.close()
            raise

    def _read_state(self):
        try:
            document = json.loads(self.path.read_text())
            payload = {key: document[key] for key in ('version', 'stream', 'reserved_through')}
            if payload['version'] != 1 or payload['stream'] != self.stream:
                raise ValueError('state version or stream identity differs')
            if document['sha256'] != hashlib.sha256(_canonical(payload)).hexdigest():
                raise ValueError('state checksum differs')
            return uint32(payload['reserved_through'], 'reserved_through')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SequenceStateError(f'Cannot trust sequence state {self.path}; refusing to reset: {exc}') from exc

    def _write_state(self, high):
        payload = {'version': 1, 'stream': self.stream, 'reserved_through': high}
        document = dict(payload, sha256=hashlib.sha256(_canonical(payload)).hexdigest())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=self.directory, prefix='.'+self.path.stem+'-', delete=False) as output:
                temporary = Path(output.name)
                output.write(json.dumps(document, sort_keys=True) + '\n')
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            _sync_directory(self.directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _reserve(self):
        if self._reserved >= MAX_SEQUENCE:
            self._failed = True
            raise SequenceStateError('uint32 sequence space exhausted; coordinate a receiver reset before reinitializing')
        high = min(MAX_SEQUENCE, self._reserved + self.block_size)
        try:
            self._write_state(high)
        except (OSError, ValueError) as exc:
            self._failed = True
            raise SequenceStateError(f'Could not durably reserve sequence range: {exc}') from exc
        self._reserved = high

    @property
    def current(self):
        with self._mutex:
            return self._current

    def next(self):
        with self._mutex:
            if self._closed or self._failed:
                raise SequenceStateError('Sequence allocator is closed or failed; no publication is permitted')
            if self._current == self._reserved:
                self._reserve()
            self._current += 1
            return self._current

    def snapshot(self):
        with self._mutex:
            return {'state_file': str(self.path), 'stream': self.stream, 'current': self._current,
                    'reserved_through': self._reserved, 'block_size': self.block_size,
                    'failed': self._failed, 'closed': self._closed}

    def close(self):
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            if self._fd is not None:
                # Closing also releases flock on normal exit; the kernel does so after SIGKILL.
                os.close(self._fd)
                self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
