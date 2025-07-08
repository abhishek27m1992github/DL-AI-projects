![Build Status](https://github.com/abhishek27m1992github/DL-AI-projects/actions/workflows/MNIST_25K_Parms-ci.yml/badge.svg)

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

---

## How to Fix

You need to add `matplotlib` (and any other required packages) to your `MNIST_25K_Parms/requirements.txt` file.

---

### **Steps:**

1. **Edit `MNIST_25K_Parms/requirements.txt`**  
   Add the following line if it’s not already present:
   ```
   matplotlib
   ```

   Also, make sure any other packages you use (like `torch`, `numpy`, etc.) are listed.

2. **Commit and push the change:**
   ```bash
   git add MNIST_25K_Parms/requirements.txt
   git commit -m "Add matplotlib to requirements.txt"
   git push origin main
   ```

3. **GitHub Actions will automatically re-run**  
   The workflow will now install `matplotlib` and the error should be resolved.

---

**Summary:**  
- Add `matplotlib` to your `requirements.txt`
- Commit and push
- The workflow will succeed if all dependencies are listed

Let me know if you need help updating your requirements file or if you see any new errors! 