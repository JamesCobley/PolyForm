name: Publish to PyPI

# Publishes on every pushed tag that looks like a version, e.g. v0.1.0
on:
  push:
    tags:
      - "v*"

jobs:
  build:
    name: Build distributions
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Build sdist and wheel
        run: |
          python -m pip install --upgrade build
          python -m build

      - name: Check metadata
        run: |
          python -m pip install --upgrade twine
          python -m twine check dist/*

      - uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    name: Publish to PyPI
    needs: build
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write   # required for Trusted Publishing (OIDC); no API token needed
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish
        uses: pypa/gh-action-pypi-publish@release/v1
