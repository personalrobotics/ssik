# Conda-forge submission instructions

The recipe in `meta.yaml` is ready to PR into `conda-forge/staged-recipes`. Steps:

## 1. Fork conda-forge/staged-recipes

```bash
gh repo fork conda-forge/staged-recipes --clone
cd staged-recipes
```

Or via the web UI: https://github.com/conda-forge/staged-recipes — "Fork".

## 2. Add the recipe

```bash
git checkout -b add-ssik
mkdir -p recipes/ssik
cp /Users/siddh/code/ikfastpy/outreach/conda-forge/meta.yaml recipes/ssik/meta.yaml
git add recipes/ssik/meta.yaml
```

## 3. Open the PR

```bash
git commit -m "Add ssik recipe"
git push --set-upstream origin add-ssik
gh pr create \
  --repo conda-forge/staged-recipes \
  --title "Add ssik" \
  --body "$(cat <<'EOF'
Adds [ssik](https://github.com/personalrobotics/ssik), a Python library for analytical inverse kinematics on 6R and 7R revolute robot arms. v1.1.0 on PyPI.

### Checklist

- [x] Title of this PR is meaningful: e.g. "Adding my_nifty_package", not "updated meta.yaml".
- [x] License (BSD-3-Clause) and license-file are correct.
- [x] Source is from official source (PyPI sdist).
- [x] Package does not vendor outdated copies of conda-forge packages.
- [x] If the package has compiled extensions: includes \`{{ compiler('c') }}\` in build requirements.
- [x] No \`noarch: python\` because the package has Cython-compiled extensions.

### Maintainer

- @siddhss5
EOF
)"
```

## 4. Respond to the CI bot

Conda-forge's `linter-action` and `boa` will comment with any recipe issues — common ones:

- **Pinning style**: conda-forge prefers `python` (no version) in `host` + `run`, with the Python version range expressed only via `skip:` in `build`. Already correct in `meta.yaml`.
- **`numpy` pinning**: must use `{{ pin_compatible('numpy') }}` in `run`. Already correct.
- **`cython` in `run`**: conda-forge usually objects to runtime-cython for sdists with prebuilt extensions, but ssik's `pyproject.toml` lists cython as a runtime dependency because compiled-source modules `import cython` at module top (see the comment in `pyproject.toml` referring to issue #144). Be prepared to explain this in the PR thread; if the reviewer pushes back, the alternative is to gate the `import cython` lines on `try/except` and drop the runtime dep.
- **License classifier**: the PEP 639 SPDX `license = "BSD-3-Clause"` in `pyproject.toml` (not the legacy classifier) — conda-forge handles this fine, but a reviewer may ask. It's correct.

## 5. After merge

Once `conda-forge/staged-recipes` merges the PR, the conda-forge bots:

1. Create `conda-forge/ssik-feedstock` (a new repo).
2. Build wheels for Linux x86_64, macOS arm64, macOS x86_64, Windows x86_64 × py3.11/3.12/3.13.
3. Push them to anaconda.org/conda-forge.

Users can then:

```bash
conda install -c conda-forge ssik
mamba install -c conda-forge ssik
```

## 6. After conda-forge package is live

Open a PR on `personalrobotics/ssik`:

- Add a conda-forge badge to `README.md` next to the PyPI / Python / License badges.
- Add a `conda install -c conda-forge ssik` line to the install section.

## 7. Future version bumps

The conda-forge `regro-cf-autotick-bot` watches PyPI. When a new ssik version is published to PyPI, the bot auto-opens a feedstock PR with the version + sha256 updated. We just need to:

- Verify the bot's PR builds clean on CI.
- Merge.

If a release changes runtime deps (new package required, version bump beyond the existing pin), the bot's PR will need manual edits to `recipes/ssik/meta.yaml` before merge.

## Notes

- The `outreach/conda-forge/meta.yaml` here is the **draft** of what gets committed to `staged-recipes/recipes/ssik/meta.yaml`. After merge, the canonical recipe lives in `conda-forge/ssik-feedstock`, and updates happen there.
- Review SLA on staged-recipes is typically ~1 week for first-time submissions. Subsequent feedstock bumps merge in hours.
