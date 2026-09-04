import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest


_spec = importlib.util.spec_from_file_location(
    "check_release_versions",
    Path(__file__).resolve().parents[1] / "scripts" / "check_release_versions.py",
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize("wheel_version,sdist_version,valid", [
    ("1.0.3", "1.0.3", True),
    ("1.0.2", "1.0.3", False),
    ("1.0.3", "1.0.3.dev1", False),
    ("1.0.3+local", "1.0.3", False),
])
def test_distribution_versions(tmp_path, wheel_version, sdist_version, valid):
    with zipfile.ZipFile(tmp_path / "specbox.whl", "w") as archive:
        archive.writestr("specbox.dist-info/METADATA",
                         f"Name: specbox\nVersion: {wheel_version}\n")
    payload = f"Name: specbox\nVersion: {sdist_version}\n".encode()
    with tarfile.open(tmp_path / "specbox.tar.gz", "w:gz") as archive:
        info = tarfile.TarInfo("specbox/PKG-INFO")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    if valid:
        _module.check_versions(tmp_path, "v1.0.3")
    else:
        with pytest.raises(ValueError):
            _module.check_versions(tmp_path, "v1.0.3")
