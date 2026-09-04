# Releases

## 1.0.3

- Add exclusive emission-line markers for H-alpha, [O III] 5008, [O II] 3728,
  Mg II, and [S III] 9533.2, with Off/Escape and spin-box exit behavior.
- Increase toolbar and marker control heights for readable desktop layouts.
- Store cumulative recovery snapshots under `temp/<sample>/vi_temp_<count>.csv`
  and preserve inspection navigation when snapshot writes fail.
- Reuse prepared observed-spectrum data when redrawing templates.
- Support the Euclid parquet flux scale and missing uncertainty inputs,
  and pair dual arms by source identity with single-arm coverage retained.
- Install `pyarrow` by default, and provide a teamwork inspection guide with
  pip upgrades, save/resume limitations, and CSV handoff instructions.

## Maintainer checklist

Versions come from Git tags through setuptools-scm. Do not edit `__version__`.
The existing `v1.0.2` tag must remain unchanged.

1. Review the changes and run `python -m pytest -q` with
   `QT_QPA_PLATFORM=offscreen` on a headless machine. Build the documentation
   with `python -m sphinx -b html specbox/docs <fresh-docs-directory>` and
   inspect the rendered participant workflow.
2. Build into a fresh directory with `python -m build --outdir <fresh-dist-directory>`.
   Run `python -m twine check <fresh-dist-directory>/*`. Inspect wheel metadata,
   dependencies, entry points, and packaged templates, and smoke-test parquet
   loading and the GUI using the built wheel.
3. Commit the validated work, then create and push the new `v1.0.3` tag on that
   commit. Untagged development builds correctly retain a development/local
   version; they are not the release artifacts.
4. Build from the clean tag and run
   `python scripts/check_release_versions.py <fresh-dist-directory> v1.0.3`.
   This requires a stable 1.0.3 version in both wheel and sdist metadata.
5. Confirm PyPI Trusted Publishing is configured for `rudolffu/specbox`, workflow
   `python-publish.yml`, environment `pypi`. Publish a GitHub Release for the
   tag using the notes above. The workflow tests and validates artifacts before
   uploading; pushing a tag alone does not upload to PyPI.
6. Confirm 1.0.3 and both distributions appear on PyPI. Verify an upgrade with
   `python -m pip install --upgrade specbox` and `python -m pip show specbox`.
   Check `specbox-viewer --help` and one batch GUI before inviting participants.
7. Check that Read the Docs has built the updated tutorial. The `latest` docs
   track Git and may describe features ahead of the published PyPI package.

Release publication and uploads are maintainer actions, separate from preparing
these files. No credentials or participant data belong in distributions.
