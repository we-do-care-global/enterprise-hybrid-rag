# Zenodo trigger — disabled pending token

## Why
The `zenodo.yml` workflow originally ran on `release: types: [published]`. During the
org -> account migration, releases were copied to this repository, which fired the
workflow for every release. All runs failed immediately.

## Failure mode
All runs failed at the first step, `Verify required secret is present`:

    if [ -z "$ZENODO_TOKEN" ]; then
      echo "::error::ZENODO_TOKEN is not set. ..."
      exit 1
    fi

The guard exits before any HTTP request is made, so no Zenodo API call, no draft
creation, and no publish ever happened.

## What changed
`on: release: types: [published]` was removed. The workflow now only runs on
`workflow_dispatch` (manual, with an explicit tag input).

## How to re-enable
1. Add the repository secret `ZENODO_TOKEN` (Zenodo personal access token with
   `deposit:write` and `deposit:actions` scopes).
2. Restore the trigger in `.github/workflows/zenodo.yml`:

       on:
         release:
           types: [published]
         workflow_dispatch:
           inputs:
             tag: ...

3. Verify with a manual `workflow_dispatch` run first.
