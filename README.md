# Machine Translation Project

This is a machine learning project aimed at classifying texts from the 20 News Groups dataset.

## Project Structure

The `src` directory contains the `text_classifier` package which holds the source code for the project. The `notebooks` directory contains Jupyter notebooks used for experimentation and data exploration.

## Setting Up the Project

To set up the project, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Jaime-Pinkman/machine_translation.git
    ```

2. Navigate to the project directory:

    ```bash
    cd machine_translation
    ```

3. Install the project dependencies:

    ```bash
    poetry install
    ```

This will install all the necessary dependencies listed in the `pyproject.toml` file and ensure that the `text_classifier` package is discoverable by Python.

## Running Tests

To run the tests, navigate to the `tests` directory and run:

```bash
pytest
