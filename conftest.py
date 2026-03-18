from __future__ import annotations

import pathlib
import shutil
import uuid

import pytest

_BASE_TMP = pathlib.Path('tmp')
_BASE_TMP.mkdir(exist_ok=True)

class _SimpleTmpFactory:
    def __init__(self, base: pathlib.Path) -> None:
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)

    def mktemp(self, basename: str = 'tmp', numbered: bool = True) -> pathlib.Path:
        suffix = uuid.uuid4().hex if numbered else ''
        name = f"{basename}-{suffix}" if suffix else basename
        path = self.base / name
        counter = 0
        while path.exists():
            counter += 1
            path = self.base / f"{name}-{counter}"
        path.mkdir(parents=True, exist_ok=False)
        return path

@pytest.fixture
def tmp_path_factory() -> _SimpleTmpFactory:
    return _SimpleTmpFactory(_BASE_TMP / 'pytest_manual')

@pytest.fixture
def tmp_path(tmp_path_factory: _SimpleTmpFactory):
    path = tmp_path_factory.mktemp('case', numbered=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
