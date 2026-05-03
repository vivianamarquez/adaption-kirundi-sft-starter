# adaption-kirundi-sft-starter
An open-source starter repo for improving low-resource Kirundi SFT data with Adaption and evaluating post-training outcomes.

## Environment

This project tracks its conda setup in two files:

- `environment.yml`: the editable environment spec for day-to-day development.
- `environment.lock.yml`: the exact exported package versions from the created environment.

Create the environment from scratch:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate adaption-kirundi-sft
```

Update the environment after changing `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

Recreate the exact exported environment:

```bash
conda env create -f environment.lock.yml
```

Register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name adaption-kirundi-sft --display-name "Python (adaption-kirundi-sft)"
```
