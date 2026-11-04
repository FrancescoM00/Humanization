#!/usr/bin/env python3
"""Archive local datasets, model weights, Python environment and runtime cache.

The compressed streams are split below GitHub's 100 MiB single-file limit.
This command never reads .git, executor configuration or credential directories.
It does not access the network and never modifies the original components.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {
    'data': {'directory': 'datasets', 'roots': ['data'], 'exclude': ['data/models']},
    'models': {'directory': 'model_weights', 'roots': ['data/models'], 'exclude': []},
    'venv': {'directory': 'python_environment', 'roots': ['.venv'], 'exclude': []},
    'cache': {'directory': 'runtime_cache', 'roots': ['.cache', '.pytest_cache', 'semantic_run.log'], 'exclude': []},
}


def inventory(root, roots, excluded):
    counts = {'regular_files': 0, 'regular_file_bytes': 0, 'symlinks': 0, 'directories': 0}
    def visit(path):
        relative = path.relative_to(root).as_posix()
        if any(relative == name or relative.startswith(name + '/') for name in excluded):
            return
        if path.is_symlink():
            counts['symlinks'] += 1
        elif path.is_dir():
            counts['directories'] += 1
            for child in path.iterdir():
                visit(child)
        elif path.is_file():
            counts['regular_files'] += 1
            counts['regular_file_bytes'] += path.stat().st_size
        else:
            raise ValueError(f'Unsupported special file: {relative}')
    for name in roots:
        visit(root / name)
    return counts


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def archive_component(name, root, output, part_bytes, threads, overwrite):
    spec = COMPONENTS[name]
    roots = [value for value in spec['roots'] if os.path.lexists(root / value)]
    if not roots:
        raise FileNotFoundError(f'No source roots found for {name}')
    destination = output / spec['directory']
    if destination.exists() and not overwrite:
        raise FileExistsError(f'Preserving {destination}; use --overwrite for an intentional rebuild')
    counts = inventory(root, roots, spec['exclude'])
    # Conservative bound: uncompressed bytes plus tar padding/header overhead.
    required = counts['regular_file_bytes'] + 2048 * (counts['regular_files'] + counts['directories'] + counts['symlinks']) + 64 * 2**20
    free = shutil.disk_usage(output).free
    if free < required:
        raise OSError(f'Insufficient disk for {name}: {free} free, conservative requirement {required}')
    build_parent = output / '.build'
    build_parent.mkdir(exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=name + '-', dir=build_parent))
    stream_hash = hashlib.sha256()
    parts = []
    current = None
    part_hash = None
    written = total = 0
    try:
        tar_command = ['tar', '-C', str(root), '--format=pax', '--hard-dereference', '-cf', '-']
        tar_command += ['--exclude=' + value for value in spec['exclude']]
        tar_command += roots
        with (build / 'tar.log').open('wb') as tar_log, (build / 'zstd.log').open('wb') as zstd_log:
            tar = subprocess.Popen(tar_command, stdout=subprocess.PIPE, stderr=tar_log)
            compressor = subprocess.Popen(['zstd', '-3', f'-T{threads}', '-c'], stdin=tar.stdout,
                                          stdout=subprocess.PIPE, stderr=zstd_log)
            tar.stdout.close()
            try:
                while True:
                    block = compressor.stdout.read(1024 * 1024)
                    if not block:
                        break
                    stream_hash.update(block)
                    total += len(block)
                    offset = 0
                    while offset < len(block):
                        if current is None:
                            filename = f'{spec["directory"]}.tar.zst.part{len(parts) + 1:04d}'
                            current = (build / filename).open('xb')
                            part_hash = hashlib.sha256()
                            written = 0
                        piece = block[offset:offset + min(part_bytes - written, len(block) - offset)]
                        current.write(piece)
                        part_hash.update(piece)
                        written += len(piece)
                        offset += len(piece)
                        if written == part_bytes:
                            current.close()
                            parts.append({'path': f'{spec["directory"]}/{current.name.rsplit("/", 1)[-1]}',
                                          'bytes': written, 'sha256': part_hash.hexdigest()})
                            current = None
                if current is not None:
                    current.close()
                    parts.append({'path': f'{spec["directory"]}/{Path(current.name).name}',
                                  'bytes': written, 'sha256': part_hash.hexdigest()})
                    current = None
                compressor_result = compressor.wait()
                tar_result = tar.wait()
                if compressor_result or tar_result:
                    raise RuntimeError(f'tar exit={tar_result}; zstd exit={compressor_result}; inspect {build}')
            except BaseException:
                if current is not None:
                    current.close()
                for process in (compressor, tar):
                    if process.poll() is None:
                        process.terminate()
                for process in (compressor, tar):
                    process.wait()
                raise
        result = {'roots': roots, 'excluded_roots': spec['exclude'], 'source_inventory': counts,
                  'compression': {'format': 'tar.zst', 'zstd_level': 3, 'threads': threads},
                  'compressed_archive_bytes': total, 'compressed_archive_sha256': stream_hash.hexdigest(),
                  'part_size_limit_bytes': part_bytes, 'parts': parts,
                  'created_utc': datetime.now(timezone.utc).isoformat()}
        atomic_json(build / 'component_manifest.json', result)
        (build / 'tar.log').unlink()
        (build / 'zstd.log').unlink()
        old = None
        if destination.exists():
            old = output / ('.old-' + spec['directory'])
            if old.exists():
                raise FileExistsError(f'Preserving previous recovery directory {old}')
            destination.rename(old)
        try:
            build.rename(destination)
        except BaseException:
            if old is not None:
                old.rename(destination)
            raise
        if old is not None:
            shutil.rmtree(old)
        return result
    finally:
        # Keep failed-build evidence; successful builds have already moved.
        if build_parent.exists() and not any(build_parent.iterdir()):
            build_parent.rmdir()


def push_groups(components, max_bytes=450 * 2**20):
    groups, current, total = [], [], 0
    for key, component in components.items():
        for part in component['parts']:
            if current and total + part['bytes'] > max_bytes:
                groups.append({'bytes': total, 'parts': current})
                current, total = [], 0
            current.append(part['path'])
            total += part['bytes']
    if current:
        groups.append({'bytes': total, 'parts': current})
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'backups')
    parser.add_argument('--component', choices=['all', *COMPONENTS], default='all')
    parser.add_argument('--part-mib', type=int, default=90)
    parser.add_argument('--threads', type=int, default=3)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.part_mib <= 90 or not 1 <= args.threads <= 3:
        parser.error('Parts must be 1..90 MiB; threads 1..3')
    for program in ['tar', 'zstd']:
        if shutil.which(program) is None:
            raise RuntimeError(f'Install the {program} executable first')
    args.root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'format_version': 1, 'source_root': str(args.root),
        'platform': {'system': platform.system(), 'machine': platform.machine(),
                     'python': platform.python_version(), 'libc': platform.libc_ver()},
        'notes': ['Includes project data, local model weights, .venv and project runtime caches.',
                  'Excludes .git and executor credentials/configuration; source and results are regular repository files.',
                  'The Linux x86_64 virtual environment is a preservation snapshot, not portable to other OS/architectures.',
                  'Its Python symlink and script shebangs reference absolute paths of the original cloud runtime.'],
        'components': {}}
    names = list(COMPONENTS) if args.component == 'all' else [args.component]
    for name in names:
        print(f'Archiving {name}...', flush=True)
        result = archive_component(name, args.root, args.output, args.part_mib * 2**20,
                                   args.threads, args.overwrite)
        manifest['components'][name] = result
        manifest['suggested_git_push_groups'] = push_groups(manifest['components'])
        atomic_json(manifest_path, manifest)
        print(json.dumps({'component': name, 'source': result['source_inventory'],
                          'parts': len(result['parts']), 'compressed_bytes': result['compressed_archive_bytes']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
