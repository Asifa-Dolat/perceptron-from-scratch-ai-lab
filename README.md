# Perceptron Learning Algorithm (From Scratch)

## Project Overview
This project implements a single-layer Perceptron model from scratch in Python. The perceptron is a simple supervised learning algorithm used for binary classification tasks.

Instead of using machine learning libraries, the model is built using basic Python constructs to demonstrate the internal working of weight updates, bias adjustment, and activation function.

---

## Features
- Single-layer perceptron implementation
- Manual weight and bias initialization
- Step activation function
- Training using gradient-based weight updates
- Works on simple logic gates (AND example used)

---

## Dataset Used
The model is trained on the AND gate dataset:

| Input 1 | Input 2 | Output |
|--------|--------|--------|
| 0      | 0      | 0      |
| 0      | 1      | 0      |
| 1      | 0      | 0      |
| 1      | 1      | 1      |

---

## How It Works
1. Inputs are multiplied with weights
2. Bias is added to the sum
3. Activation function determines output (0 or 1)
4. Error is calculated
5. Weights and bias are updated iteratively

---

## Tech Stack
- Python (No external ML libraries)
- Jupyter Notebook / Python Script

---

## Learning Outcome
- Understanding of neural network basics
- How perceptron learns using weight updates
- Concept of linear separability

---

