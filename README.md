# Efficient Progressive Training for Identity-Consistent Face Swapping

This repository contains the official implementation for our ICASSP 2026 paper:

Efficient Progressive Training Framework for Identity-Consistent Face Swapping


## Environment

Python dependencies are exported in [requirements.txt](requirements.txt).

Typical setup:

1. Create and activate a clean Python environment.
2. Install dependencies:

	pip install -r requirements.txt

3. Verify your local CUDA, PyTorch, and driver compatibility.

## Project Structure

- [examples/text_to_image](examples/text_to_image): training scripts, models, pipelines, losses, and dataset utilities.

## Notes

- Paths to datasets and checkpoints are expected to be provided via script arguments.
- Please avoid committing private paths, credentials, or personal experiment artifacts.
- License: see [LICENSE](LICENSE).

## Acknowledgements

This project builds on top of the Hugging Face Diffusers ecosystem.
We sincerely thank the Diffusers team and contributors for their high-quality open-source work.

## Citation

If you find this repository useful, please cite our paper. A BibTeX entry will be added after publication metadata is finalized.
