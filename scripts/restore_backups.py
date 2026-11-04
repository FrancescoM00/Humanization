#!/usr/bin/env python3
"""Verify and restore selected workspace snapshots without network access.

Archives are streamed into a fresh staging directory. Symlinks are created only
after files have been written, so extraction never follows archive symlinks.
Existing component roots are preserved unless --overwrite is explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]


def part_paths(backup_dir, component):
    paths = []
    archive_hash = hashlib.sha256()
    archive_bytes = 0
    for item in component['parts']:
        relative = PurePosixPath(item['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in item['path']:
            raise ValueError('Unsafe part path in manifest')
        path = (backup_dir / item['path']).resolve()
        if not path.is_relative_to(backup_dir.resolve()):
            raise ValueError('Backup part points outside the backup directory')
        digest = hashlib.sha256()
        count = 0
        with path.open('rb') as handle:
            if handle.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise RuntimeError('This is a Git LFS pointer, not archive data. Follow backups/README.md: git lfs install; git fetch origin cloud-backup; git lfs fetch origin FETCH_HEAD --include="backups/**"; git restore --source=FETCH_HEAD --worktree -- backups/.')
            handle.seek(0)
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
                archive_hash.update(block)
                count += len(block)
        if count != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f'Backup part failed verification: {path.name}')
        archive_bytes += count
        paths.append(path)
    if archive_bytes != component['compressed_archive_bytes'] or archive_hash.hexdigest() != component['compressed_archive_sha256']:
        raise ValueError('Concatenated archive failed verification')
    return paths


def member_path(member, component):
    name = member.name
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name:
        raise ValueError(f'Unsafe archive member: {name!r}')
    normalized = path.as_posix().rstrip('/')
    if not any(normalized == root or normalized.startswith(root + '/') for root in component['roots']):
        raise ValueError(f'Archive member outside registered roots: {name!r}')
    if any(normalized == root or normalized.startswith(root + '/') for root in component.get('excluded_roots', [])):
        raise ValueError(f'Archive member in an excluded root: {name!r}')
    if not (member.isfile() or member.isdir() or member.issym()):
        raise ValueError(f'Unsupported archive member type: {name!r}')
    return Path(*path.parts)


def unpack(paths, component, stage=None):
    errors = []
    with tempfile.TemporaryFile() as error_log:
        process = subprocess.Popen(['zstd', '-d', '-c'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=error_log)
        def feed():
            try:
                with process.stdin:
                    for path in paths:
                        with path.open('rb') as handle:
                            shutil.copyfileobj(handle, process.stdin, length=1024 * 1024)
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=feed, daemon=True)
        worker.start()
        directories, symlinks = [], []
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    relative = member_path(member, component)
                    if stage is None:
                        print(member.name)
                        continue
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if member.isdir():
                        target.mkdir(exist_ok=True)
                        directories.append((target, member.mode & 0o777, member.mtime))
                    elif member.issym():
                        symlinks.append((target, member.linkname))
                    else:
                        with archive.extractfile(member) as source, target.open('xb') as destination:
                            shutil.copyfileobj(source, destination, length=1024 * 1024)
                        target.chmod(member.mode & 0o777)
                        os.utime(target, (member.mtime, member.mtime))
            # Drain tar padding so the compressor writer finishes normally.
            while process.stdout.read(1024 * 1024):
                pass
            worker.join()
            status = process.wait()
            if status or errors:
                raise RuntimeError(f'zstd exit={status}; input errors={errors}')
            for target, link in symlinks:
                if os.path.lexists(target):
                    raise ValueError(f'Conflicting symlink in archive: {target}')
                target.symlink_to(link)
            for target, mode, mtime in sorted(directories, key=lambda item: len(item[0].parts), reverse=True):
                target.chmod(mode)
                os.utime(target, (mtime, mtime))
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.stdout.close()
            worker.join()
            process.wait()
            raise


def remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-dir', type=Path, default=ROOT / 'backups')
    parser.add_argument('--root', type=Path, default=ROOT, help='Destination project root')
    parser.add_argument('--component', choices=['all', 'data', 'models', 'venv', 'cache'], default='all')
    parser.add_argument('--list', action='store_true', help='Verify archives and list their members; write no component files')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing selected component roots after successful staged extraction')
    args = parser.parse_args()
    if shutil.which('zstd') is None:
        raise RuntimeError('Install the zstd executable before restoring')
    manifest = json.loads((args.backup_dir / 'manifest.json').read_text())
    if manifest['format_version'] != 1:
        raise ValueError('Unsupported backup format')
    names = list(manifest['components']) if args.component == 'all' else [args.component]
    components = [manifest['components'][name] for name in names]
    roots = sorted({root for component in components for root in component['roots']}, key=lambda value: len(PurePosixPath(value).parts))
    # A complete data restore already owns its nested data/models destination.
    minimal = [name for name in roots if not any(name.startswith(other + '/') for other in roots if other != name)]
    args.root = args.root.resolve()
    for root in minimal:
        path = PurePosixPath(root)
        if path.is_absolute() or '..' in path.parts or '\\' in root:
            raise ValueError('Unsafe destination root in manifest')
        if not args.list and os.path.lexists(args.root / root) and not args.overwrite:
            raise FileExistsError(f'Preserving {args.root / root}; restore into an empty --root or use --overwrite explicitly')
    verified = [part_paths(args.backup_dir, component) for component in components]
    if args.list:
        for paths, component in zip(verified, components):
            unpack(paths, component)
        return
    args.root.mkdir(parents=True, exist_ok=True)
    required = sum(component['source_inventory']['regular_file_bytes'] for component in components) + 128 * 2**20
    if shutil.disk_usage(args.root).free < required:
        raise OSError(f'Insufficient space: restoration requires approximately {required} bytes')
    with tempfile.TemporaryDirectory(prefix='.restore-stage-', dir=args.root) as stage_name:
        stage = Path(stage_name)
        for name, paths, component in zip(names, verified, components):
            print(f'Restoring {name} into staging...', flush=True)
            unpack(paths, component, stage)
        with tempfile.TemporaryDirectory(prefix='.restore-previous-', dir=args.root) as previous_name:
            previous = Path(previous_name)
            written, moved = [], []
            try:
                for root in minimal:
                    destination, source = args.root / root, stage / root
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if os.path.lexists(destination):
                        old = previous / root
                        old.parent.mkdir(parents=True, exist_ok=True)
                        destination.rename(old)
                        moved.append(root)
                    source.rename(destination)
                    written.append(root)
            except BaseException:
                for root in reversed(written):
                    remove_path(args.root / root)
                for root in reversed(moved):
                    (previous / root).rename(args.root / root)
                raise
    print('Restoration complete. The saved virtual environment may need recreation on this OS/path.')


if __name__ == '__main__':
    main()
