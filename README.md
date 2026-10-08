
# 🏥 Early Detection of Chronic Kidney Disease Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?style=for-the-badge&logo=sqlite)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![HTML](https://img.shields.io/badge/HTML5-Frontend-red?style=for-the-badge&logo=html5)

> An AI-powered web application that predicts the risk of Chronic Kidney Disease (CKD) using Machine Learning — designed for the **general public** with no lab tests required.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [ML Models Used](#ml-models-used)
- [How It Works](#how-it-works)
- [Installation & Setup](#installation--setup)
- [Running the App](#running-the-app)
- [API Endpoints](#api-endpoints)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)

---

## 📖 About the Project

Chronic Kidney Disease (CKD) affects over **850 million people worldwide** and often shows **no symptoms** until 60–70% of kidney function is already lost. Early detection is critical.

This project uses **Machine Learning** trained on the UCI/Kaggle CKD dataset to predict CKD risk. Since our target users are the **general public** (without lab test reports), we built a smart **Symptom-to-Clinical Mapping Layer** that converts simple yes/no symptom answers into approximate clinical values — which are then fed into the ML model.

---

## ✨ Features

- 🔐 **User Authentication** — Register & Login with hashed passwords
- 📝 **11 Simple Questions** — No lab tests needed, anyone can answer
- 🤖 **ML Prediction** — Random Forest model with 95%+ accuracy
- 📊 **Risk Assessment** — Low / Medium / High risk with probability score
- 💾 **SQLite Database** — All users and predictions saved permanently
- 📓 **Research Notebook** — Explains CKD, biomarkers, and ML methodology
- 📱 **Responsive Design** — Works on desktop and mobile
- 🎨 **Professional UI** — Dark theme with glassmorphism design

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Backend** | Python, Flask |
| **Database** | SQLite (via Flask-SQLAlchemy) |
| **ML Model** | scikit-learn (Random Forest, SVM, KNN, Logistic Regression) |
| **Security** | Werkzeug password hashing |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## 📁 Project Structure

```
CKD_DETECT_PROJECT/
│
├── Codes/
│   ├── app_code.py          # Flask backend (API + database + ML)
│   ├── ckd-app-v3.html      # Frontend (single-page app)
│   ├── requirements.txt     # Python dependencies
│   ├── .gitignore           # Files excluded from GitHub
│   ├── .env.example         # Environment variable template
│   │
│   ├── ckd_model.pkl        # Trained ML model
│   ├── ckd_scaler.pkl       # Feature scaler
│   ├── ckd_features.pkl     # Feature list 
│   └── ckd_users.db         # SQLite database (not on GitHub)
│
├── Spl_project Notebook.ipynb   # Jupyter notebook (full ML pipeline)
└── README.md
```

---

## 📊 Dataset

- **Source:** [Kaggle CKD Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease) (UCI Repository)
- **Records:** 400 patient records
- **Features:** 24 clinical attributes
- **Target:** Binary — `ckd` / `notckd`

### Key Features in Dataset

| Type | Features |
|---|---|
| **Numeric** | Age, Blood Pressure, Blood Glucose, Blood Urea, Serum Creatinine, Sodium, Potassium, Hemoglobin, Packed Cell Volume, WBC Count, RBC Count |
| **Categorical** | Red Blood Cells, Pus Cells, Bacteria, Hypertension, Diabetes, Coronary Artery Disease, Appetite, Pedal Edema, Anemia |

---

## 🤖 ML Models Used

| Model | Accuracy | AUC-ROC |
|---|---|---|
| **Random Forest** ⭐ | 98.33% | 0.99 |
| SVM | 97.50% | 0.98 |
| Gradient Boosting | 97.10% | 0.98 |
| Logistic Regression | 96.67% | 0.97 |
| KNN | 94.17% | 0.96 |

> ✅ **Random Forest** was selected as the final model for its highest accuracy and best interpretability.

---

## ⚙️ How It Works

```
User answers 11 symptom questions
            ↓
Symptom-to-Clinical Mapping Layer
(converts answers to approximate clinical values)
            ↓
24 clinical features built
            ↓
StandardScaler normalizes the features
            ↓
Random Forest model predicts CKD risk
            ↓
Result shown: Low / Medium / High risk + recommendation
            ↓
Result saved to SQLite database
```

### Symptom Questions Asked
1. 🎂 Age
2. ❤️ High Blood Pressure (Hypertension)?
3. 🩸 Diabetes?
4. 😴 Unusual fatigue or tiredness?
5. 🦵 Swelling in feet or ankles?
6. 🍽️ Poor appetite or nausea?
7. 🚽 Changes in urination frequency?
8. 😣 Persistent back or side pain?
9. 👨‍👩‍👦 Family history of kidney disease?
10. 🚬 Smoking or alcohol use?
11. ⚖️ Overweight or obese?

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ckd-detection.git
cd ckd-detection/Codes
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Generate ML model files
> The `.pkl` files are not included in the repository for size reasons.
> Run the Jupyter notebook to generate them:

1. Open `Spl_project Notebook.ipynb` in Google Colab or Jupyter
2. Upload `kidney_disease.csv` from [Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease)
3. Run all cells
4. Download `ckd_model.pkl`, `ckd_scaler.pkl`, `ckd_features.pkl`
5. Place them in the `Codes/` folder

### Step 4 — Run the app
```bash
python app_code.py
```

### Step 5 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend HTML |
| `POST` | `/api/register` | Register a new user |
| `POST` | `/api/login` | Login existing user |
| `POST` | `/api/predict` | Run CKD prediction |
| `GET` | `/api/users` | View all registered users |
| `GET` | `/api/history/<email>` | View prediction history for a user |

### Example — Register
```json
POST /api/register
{
  "username": "john_doe",
  "name": "John Doe",
  "email": "john@email.com",
  "phone": "9876543210",
  "password": "mypassword"
}
```

### Example — Predict
```json
POST /api/predict
{
  "age": 55,
  "hypertension": "yes",
  "diabetes": "yes",
  "fatigue": "yes",
  "swelling": "yes",
  "poor_appetite": "no",
  "urination": "less",
  "back_pain": "no",
  "family_history": "yes",
  "smoke_alcohol": "no",
  "obesity": "no"
}
```

---

## 🔮 Future Improvements

- [ ] Deploy on Render / Railway (live URL)
- [ ] Add JWT token authentication
- [ ] User dashboard showing prediction history
- [ ] Mobile app version
- [ ] More symptom questions for better accuracy
- [ ] Doctor login with full clinical parameter input
- [ ] Email alerts for high-risk users
- [ ] Multi-language support

---

## ⚠️ Disclaimer

> This application is developed for **educational and academic purposes only**.
> It is **NOT a medical diagnostic tool** and should **NOT replace professional medical consultation**.
> If you are concerned about your kidney health, please consult a qualified nephrologist or doctor.

---

## 👩‍💻 Author

**Saladi Vanshika**
- Project: Special Project — CKD Early Detection


---

## 📄 License

This project is for academic use only. All rights reserved.

---

<p align="center">Made with ❤️ for early kidney health awareness</p>
