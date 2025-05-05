# 🚗 Vehicle Sales Predictor

Predict future vehicle sales like a pro.
> This open-source project demonstrates how to build, track, and deploy a state-of-the-art machine learning pipeline — from raw data to actionable predictions. It uses modern MLOps tools like MLflow, DVC, and GitHub for reproducibility and collaboration.


## ✨ Features

- 🚀 End-to-End Pipeline: From raw data to predictions

- 🔄 MLOps: Track experiments with MLflow, version data with DVC, and sync code with Git

- 🌟 SOTA Model: Tuned XGBoost delivering high performance, adaptable to any tabular data project

- 🧠 Robust Feature Engineering: Industry-grade preprocessing & encoding practices

- 📈 Production-Ready: Modular design for training, inference, and deployment


## 🛠️ Setup
```shell
# Clone the repo
git clone https://github.com/hongyingyue/vehicle-sales-predictor.git
cd vehicle-sales-predictor

# Set up your virtual environment (recommended)
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install -r requirements.txt
```


## 🚀 Getting Started

Train your model:
```shell
cd examples
python run_train.py
```

Make prediction server with the trained model:
```shell
python app.py
```

Track your experiments
```
mlflow ui
```


## Experiments
