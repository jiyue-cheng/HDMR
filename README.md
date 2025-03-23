# HDMR Project

This project implements a model using High Dimensional Model Representation (HDMR) to approximate a target function using neural networks.

## Overview

The goal of this project is to create a model that approximates a target function using HDMR techniques. The model is trained with randomly generated data, and it includes zero, first, second, and third-order modules. The network learns the contribution of each variable and their interactions with others, providing sensitivity measures over time.

## Requirements

To run this project, the following libraries are required:

- `torch` (for PyTorch)
- `matplotlib` (for plotting)
- `numpy` (optional for other operations)

You can install the dependencies using pip:

```bash
pip install torch matplotlib numpy
