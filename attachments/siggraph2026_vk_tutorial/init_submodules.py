# Copyright (c) 2026, Khronos Group and contributors
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(args):
    return subprocess.run(args, cwd=ROOT, text=True, check=True, capture_output=True)


def git_config(key):
    result = subprocess.run(
        ["git", "config", "--file", ".gitmodules", "--get", key],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Missing .gitmodules key: {key}")
    return result.stdout.strip()


def git_config_optional(key):
    result = subprocess.run(
        ["git", "config", "--file", ".gitmodules", "--get", key],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value else None


def submodule_names():
    paths = run([
        "git",
        "config",
        "--file",
        ".gitmodules",
        "--get-regexp",
        r"^submodule\..*\.path$",
    ]).stdout.splitlines()

    names = []
    for line in paths:
        key, _ = line.split(maxsplit=1)
        names.append(key.removeprefix("submodule.").removesuffix(".path"))
    return names


def normalize_sparse_path(path):
    return path.replace("\\", "/").strip().strip("/")


def current_sparse_checkout_paths(path):
    result = subprocess.run(
        ["git", "-C", path, "sparse-checkout", "list"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return [normalize_sparse_path(line) for line in result.stdout.splitlines() if line.strip()]


def configure_sparse_checkout(path, sparse_path):
    desired_path = normalize_sparse_path(sparse_path)
    current_paths = current_sparse_checkout_paths(path)
    if current_paths is not None and desired_path in current_paths:
        print(f"Sparse checkout for {path} already configured: {sparse_path}")
        return False

    print(f"Configuring sparse checkout for {path}: {sparse_path}")
    run(["git", "-C", path, "sparse-checkout", "set", sparse_path])
    return True


def ensure_sparse_submodule(name, path, url, commit, sparse_path):
    target = ROOT / path
    git_dir = target / ".git"

    if target.exists():
        if not git_dir.exists():
            raise RuntimeError(f"{path} exists but is not a Git checkout or submodule")
        print(f"Submodule {name} exists at {path}")
        configure_sparse_checkout(path, sparse_path)
        return False

    print(f"Adding sparse submodule {name} at {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", url, path])
    configure_sparse_checkout(path, sparse_path)
    run(["git", "-C", path, "checkout", commit])
    run(["git", "submodule", "add", "--force", "--name", name, url, path])
    run(["git", "submodule", "absorbgitdirs", path])
    return True


def ensure_submodule(name, path, url, commit, sparse_path=None):
    if sparse_path:
        return ensure_sparse_submodule(name, path, url, commit, sparse_path)

    target = ROOT / path
    git_dir = target / ".git"

    if target.exists():
        if not git_dir.exists():
            raise RuntimeError(f"{path} exists but is not a Git checkout or submodule")
        print(f"Submodule {name} exists at {path}")
        return False

    print(f"Adding submodule {name} at {path}")
    run(["git", "submodule", "add", "--force", "--name", name, url, path])
    return True


def main():
    if not (ROOT / ".git").exists():
        run(["git", "init"])
    else:
        print(f"Skippingg git initalization. {ROOT}.git folder already exists")

    names = submodule_names()
    if not names:
        raise RuntimeError("No submodules found in .gitmodules")

    init_submodules = False
    for name in names:
        path = git_config(f"submodule.{name}.path")
        url = git_config(f"submodule.{name}.url")
        commit = git_config(f"submodule.{name}.commit")
        sparse_path = git_config_optional(f"submodule.{name}.sparse-path")
        if not commit:
            raise RuntimeError(f"Missing commit for submodule {name}")
        if ensure_submodule(name, path, url, commit, sparse_path):
            init_submodules=True

    if init_submodules:
        print(f"initalization of git submodules at {ROOT}")
        run(["git", "-C", f"{ROOT}", "submodule", "init"])
        run(["git", "-C", f"{ROOT}", "submodule", "update", "--recursive"])

    for name in names:
        path = git_config(f"submodule.{name}.path")
        commit = git_config(f"submodule.{name}.commit")
        sparse_path = git_config_optional(f"submodule.{name}.sparse-path")
        if sparse_path:
            configure_sparse_checkout(path, sparse_path)
        current_commit = run(["git", "-C", path, "rev-parse",  "HEAD"]).stdout.strip()
        if current_commit == commit:
            print(f"Skipping {path} update. Already at correct commit")
        else:
            print(f"Updating {path} from {current_commit} to {commit}")
            run(["git", "-C", path, "checkout", commit])
            run(["git", "-C", path, "submodule", "update", "--init", "--recursive"])

if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, RuntimeError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr or error, file=sys.stderr)
            sys.exit(error.returncode)
        print(error, file=sys.stderr)
        sys.exit(1)
