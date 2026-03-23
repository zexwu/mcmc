# mcmc

Lightweight microlensing fitting tools built around `emcee`.

## Install

```bash
pip install numpy matplotlib emcee toml numba
```


## Quick start

1. Copy [conf.toml](conf.toml) and edit it for your event.
2. Put photometry files under `input`.
3. Run:

```bash
python -m mcmc.cli run conf.toml
python -m mcmc.cli lc conf.toml
python -m mcmc.cli chi2 output/chain.csv --names t0 u0 teff
```

## Config

Required top-level fields:

- `event`
- `coords = "RA DEC"`
- `input`
- `output`

Main sampler sections:

- `[mcmc]`: model name, one of `SingleLens`, `BinaryLens`, `BinaryLensOrb`
- `[mcmc.config]`: sampler settings
- `[mcmc.blobs]`: optional derived outputs such as `A_ref`
- `[mcmc.parameters]`: parameter table with `start`, `sigma`, `fixed`, and optional `bounds`
- `[[phot]]`: repeated dataset blocks

Example dataset:

```toml
[[phot]]
label = "OGLE"
filename = "ogle_I.txt"
filter = "I"
blending = true
mask_rows = []
error_scale = 1.0
error_floor = 0.0
```

Use [conf.toml](conf.toml) as the full template.

## Outputs

- `output/chain.csv`: flattened posterior samples
- `output/best.csv`: best-fit row
- `output/lc.png`: light-curve plot
