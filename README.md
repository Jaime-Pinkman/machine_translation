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

3. Add the `src` directory to your `PYTHONPATH` environment variable. This allows Python to import the project packages. Run the following command:

    ```bash
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
    ```

    This command adds the current project's `src` directory to your `PYTHONPATH`.

Please note that this setting will only persist for the duration of your current shell session. If you start a new session, you will need to run the command again.
