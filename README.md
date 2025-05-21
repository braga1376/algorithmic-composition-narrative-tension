# Algorithmic Composition Using Narrative Structure and Tension

This repository contains the code for the paper "Algorithmic Composition Using Narrative Structure and Tension" by Francisco Braga, Gilberto Bernardes, Roger B. Dannenberg, and Nuno Correia, accepted in the IJCAI 2025 AI, Arts & Creativity special track.

## Getting Started

The best way to understand the system and see it in action is to start with the main test notebook:

- [`musical_narrative_test.ipynb`](./musical_narrative_test.ipynb): This notebook provides a comprehensive overview of the system, demonstrating how narrative structures are used to generate musical pieces with varying tension levels.

## Exploring Specific Components

If you are curious about any particular part of the system, the `test_notebooks` folder contains individual notebooks for testing and demonstrating specific modules:

- [`test_notebooks/genetic_algorithm_test.ipynb`](./test_notebooks/genetic_algorithm_test.ipynb): Demonstrates the genetic algorithm used for evolving musical solutions.
- [`test_notebooks/harmony_grammar_test.ipynb`](./test_notebooks/harmony_grammar_test.ipynb): Shows how the context-free grammar generates harmonic progressions.
- [`test_notebooks/motif_composer_test.ipynb`](./test_notebooks/motif_composer_test.ipynb): Illustrates the generation of musical motifs for characters based on narrative roles.
- [`test_notebooks/tension_test.ipynb`](./test_notebooks/tension_test.ipynb): Explains and tests the tension model used to calculate musical tension.

## Project Structure

- `api_key.py`: (Anthropic's Claude API key, ensure it's gitignored if it contains sensitive information)
- `constants.py`: Defines constants used throughout the project.
- `genetic_algorithm.py`: Implements the genetic algorithm for musical composition.
- `harmony_grammar.py`: Defines the context-free grammar for generating harmonies.
- `motif_composer.py`: Contains logic for composing musical motifs.
- `musical_narrative.py`: The core module for generating musical narratives.
- `narrative_example.py`: Provides examples of narrative structures.
- `narrative_tension.py`: Manages narrative tension curves.
- `narrative.py`: Defines the narrative structure and components.
- `requirements.txt`: Lists the Python dependencies for this project.
- `utils.py`: Contains utility functions used across different modules.
- `tension_model/`: Contains the implementation of the musical tension model.
- `results_analysis/`: Contains notebooks and data for analyzing experiment results.
- `saved_data/`: Directory for storing pre-computed data or model outputs.

## Installation

To run the code in this repository, you will need Python 3.12. Make sure to install the dependencies listed in `requirements.txt`:


You might also need to set up an API key for certain functionalities (e.g., if an LLM is used for motif generation via an API). Please refer to `api_key.py` and add a key if you want to use this part of the system (currently there is None).
