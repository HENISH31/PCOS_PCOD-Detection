# PCOD Detection Web Application 💻🩺

A machine learning web app built using **Python + Flask** to predict the presence of PCOD based on clinical and lifestyle features.

## 🔍 Project Overview

This web application:
- Accepts user input via an HTML form
- Uses a trained SVM model to make predictions
- Displays results in a user-friendly format

## 💻 Tech Stack

- Python
- Flask
- HTML, CSS (Frontend)
- scikit-learn, pandas, NumPy
- Trained using SVM

## 📁 File Structure

- `app.py` – Flask server with model integration
- `pcod_model.pkl` – Trained machine learning model
- `templates/` – HTML pages (index + result)
- `static/` – CSS files (optional)
- `PCOD-10.csv` – Dataset used for training
- `requirements.txt` – Project dependencies

## 🚀 Running the App Locally

```bash
pip install -r requirements.txt
python app.py
