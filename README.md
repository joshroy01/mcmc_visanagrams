# mcmc_visanagrams

## Installation

This repository is setup so that you can `pip install` the package (with some important caveats).

### Local Install

Currently, no dependencies are specified for the package as the environment setup is quite complicated. This is (hopefully) going to change before the project is finished up!

To install the package for local use:
1. Clone the repository.
2. Create the Anaconda environment with:
    ```bash
    conda env create --name YOUR_ENV_NAME --file=environment_conda_with_torch_deepfloyd.yml
    ```
    This is a pretty long environment name. We hope to change this in the future.
3. Activate the Anaconda environment:
    ```bash
    conda activate YOUR_ENV_NAME
    ```
4. Install the `mcmc_visanagrams` as an editable `pip` package:
    ```bash
    cd /PATH/TO/REPOSITORY/ROOT/DIRECTORY
    pip install -e .
    ```

### Colab Install

It is the hope that this is setup in a way that is compatible with Colab as well as with local installs. For installation and usage examples using Colab, refer to [the ipython notebook we've been using](./prototyping/mcmc/image_tapestry.ipynb). This notebook was adapted from the MCMC repository.
