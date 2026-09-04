"""Check wheel and sdist metadata against a stable release tag."""

import argparse
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile

from packaging.version import Version


def check_versions(directory, tag):
    expected = Version(tag.removeprefix("v"))
    if expected.is_prerelease or expected.is_devrelease or expected.local:
        raise ValueError("Release tag must identify a stable public version")
    wheels = list(Path(directory).glob("*.whl"))
    sdists = list(Path(directory).glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one wheel and one sdist")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Expected one wheel METADATA file")
        wheel_metadata = archive.read(names[0])
    with tarfile.open(sdists[0]) as archive:
        names = [n for n in archive.getnames()
                 if n.endswith("/PKG-INFO") and n.count("/") == 1]
        if len(names) != 1:
            raise ValueError("Expected one top-level sdist PKG-INFO file")
        sdist_metadata = archive.extractfile(names[0]).read()
    for path, payload in zip([wheels[0], sdists[0]], [wheel_metadata, sdist_metadata]):
        metadata = BytesParser().parsebytes(payload)
        actual = Version(metadata["Version"])
        if metadata["Name"] != "specbox" or actual != expected:
            raise ValueError(f"{path.name}: expected specbox {expected}, got {actual}")
        if actual.is_devrelease or actual.local or actual.is_prerelease:
            raise ValueError(f"{path.name}: not a stable public release")
        print(f"Verified {path.name}: {actual}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("tag")
    args = parser.parse_args()
    check_versions(args.directory, args.tag)
