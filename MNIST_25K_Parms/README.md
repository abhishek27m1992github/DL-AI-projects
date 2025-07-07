[![CI/CD Pipeline](https://github.com/${{github.repository}}/actions/workflows/ci.yml/badge.svg)](https://github.com/${{github.repository}}/actions/workflows/ci.yml)

# MNIST DNN CI/CD Example

This project demonstrates a basic CI/CD pipeline for a machine learning project using a 3-layer DNN (with convolutions and a fully connected layer) on the MNIST dataset.

## Project Structure
- `train.py`: Trains a simple CNN on MNIST for 1 epoch and saves the model with a CPU and timestamp suffix.
- `test_model.py`: Runs automated tests for parameter count, input/output shape, and accuracy (>95%).
- `.github/workflows/ci.yml`: GitHub Actions workflow for CI/CD.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Ignores model files, cache, and environments.

## Local Setup
1. **Install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Train the model:**
   ```bash
   python train.py
   ```
   This will save a model file like `model_CPU_YYYYMMDD_HHMMSS.pt`.
3. **Run tests:**
   ```bash
   pytest test_model.py
   ```
   This will check:
   - Model has < 25k parameters
   - Model accepts 28x28 input and outputs 10 classes
   - Model achieves > 95% accuracy on MNIST test set

## CI/CD Pipeline (GitHub Actions)
- On every push or pull request to `main`, the workflow will:
  1. Install dependencies
  2. Train the model
  3. Run all tests
  4. Fail the build if any test fails

## Deployment
- The trained model is saved with a suffix indicating CPU and the training timestamp (e.g., `model_CPU_20230601_123456.pt`).
- You can upload this model to your deployment target as needed.

---
**Note:** All code runs on CPU and is designed for local and CI/CD (GitHub Actions) environments. 