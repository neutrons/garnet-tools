# Garnet

![Garnet logo](garnet.svg)

Garnet (Single Crystal Graphical Advanced Reduction Neutron Event Toolkit) is
a Python framework for data reduction and analysis of single-crystal neutron
diffraction experiments.

It provides:

- Automated reduction workflows for time-of-flight Laue instruments.
- Normalization, parametrization, and peak-integration workflows.
- An interactive experiment browser for navigating IPTS data.
- Support for TOPAZ and other SNS single-crystal instruments.

The project uses Mantid for neutron data processing and includes Python APIs,
command-line workflows, plotting utilities, and a graphical application.

## Installation

Garnet uses Conda for its development and runtime environment. After cloning
the repository, create the environment and install Garnet in editable mode:

```bash
git clone https://github.com/neutrons/garnet-tools.git
cd garnet-tools
conda env create -f environment.yml
conda activate garnet
python -m pip install -e .
```

If the `garnet` environment already exists, update it with:

```bash
conda env update -f environment.yml --prune
conda activate garnet
python -m pip install -e .
```

## Running Garnet

Launch the graphical application with:

```bash
garnet
```

Reduction workflows can also be run directly. For example:

```bash
python src/garnet/workflow.py <reduction-plan.yaml>
```

See the [documentation](docs/index.rst) and the tutorial guides for workflow
configuration and instrument-specific examples.

## Testing

With the `garnet` environment active, run the test suite with:

```bash
pytest
```

The GitHub Actions test job installs the package in editable mode and runs the
same test command on every push.

## Documentation

Build the API documentation locally with:

```bash
sphinx-apidoc -o docs/source src/garnet
sphinx-build -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` after the build completes. The source
documentation is organized into:

- [Project overview](docs/index.rst)
- [Reduction guides](docs/source/garnet.guides.rst)
- [Tutorials](docs/source/garnet.tutorials.rst)
- [Utilities](docs/source/garnet.utilities.rst)

Documentation deployment is configured in
`.github/workflows/gh-pages.yml`.

## Continuous integration and releases

The Conda workflow in
[`.github/workflows/python-package-conda.yml`](.github/workflows/python-package-conda.yml)
has three responsibilities:

1. Install the development environment and run the test suite.
2. Build a noarch Conda package, install it into a clean test environment,
   and verify that its Conda and Python versions match.
3. Publish packages to the `neutrons` Anaconda organization when the workflow
   runs for `main` or a version tag.

The release labels are:

- Pushes to `main` publish to the `dev` label.
- Version tags such as `v1.2.3` publish to the `main` label.
- The `dev` cleanup job retains the five newest package versions.

Publishing requires the repository or organization GitHub Actions secret
`ANACONDA_TOKEN`. Reusable Conda package actions are provided by
[`neutrons/gh-actions`](https://github.com/neutrons/gh-actions).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, pre-commit,
style, testing, and pull-request guidance.

## License

Garnet is distributed under the GPL-3.0-or-later license. See [LICENSE](LICENSE)
for the full license text.
