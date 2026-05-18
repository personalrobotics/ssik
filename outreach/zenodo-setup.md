# Zenodo DOI setup

One-time configuration so every future tagged GitHub release is auto-archived to Zenodo and assigned a DOI.

## Steps (your accounts, ~10 min)

1. **Sign in to Zenodo with GitHub OAuth.**
   - Go to https://zenodo.org/login/.
   - Click "Sign in with GitHub". Approve the OAuth scope (read repo metadata + create releases).
   - This links the Zenodo account to the `siddhss5` GitHub user. (Zenodo can archive any repo you have admin on, including org repos.)

2. **Enable Zenodo for `personalrobotics/ssik`.**
   - Go to https://zenodo.org/account/settings/github/.
   - Find `personalrobotics/ssik` in the list (you may need to click "Sync now" first, top-right).
   - Flip the toggle to **ON**.

3. **Cut a new release to trigger the first archive.**
   Zenodo only archives releases created **after** the toggle was flipped. The v1.1.0 release is not automatically backdated. Two options:
   - **Cheapest:** wait until the next release (v1.1.1 or v1.2.0) and the DOI gets minted then.
   - **If you want a DOI on v1.1.0 now:** manually re-tag — `git tag -a v1.1.0.1 -m "rebuild for Zenodo archive" && git push --tags`, let the release workflow run, and Zenodo will pick it up. Then update CITATION.cff + the README badge to point to that DOI.

4. **Once the DOI is minted, add the badge.** Zenodo shows the badge markdown on the deposit page; it looks like:
   ```markdown
   [![DOI](https://zenodo.org/badge/<repo_id>.svg)](https://zenodo.org/badge/latestdoi/<repo_id>)
   ```
   Add it to the top of `README.md` next to the PyPI / Python version badges, and update `CITATION.cff` with the concept DOI (the version-independent one):
   ```yaml
   identifiers:
     - type: doi
       value: "10.5281/zenodo.XXXXXXX"
       description: "Concept DOI — points to the latest version."
   ```

## Notes

- Zenodo mints two DOIs per release: a **concept DOI** (stable across versions, always resolves to the latest) and a **version DOI** (frozen at one release). Papers should cite the concept DOI unless they need to pin a specific version.
- Maximum deposit size is 50 GB (well under any wheel size we'd ever ship).
- The Zenodo metadata is auto-populated from `CITATION.cff` if present, which is why we landed that file first.
- Zenodo's GitHub integration documentation: https://help.zenodo.org/docs/profile/linking-github-account/.
