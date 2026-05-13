# ⚡ DataML Studio

A production-ready **CSV Data Analysis + ML Predictions** web app built with Streamlit.

## Features
- 📊 **Auto EDA** — distributions, correlations, missing value analysis
- 🧹 **Smart Preprocessing** — auto encodes, imputes, and scales data
- 🤖 **Auto ML** — trains & compares 5 models (Random Forest, Gradient Boosting, Logistic/Linear Regression, KNN, SVM/SVR)
- 📈 **Feature Importance** — bar chart for tree-based models
- 🔲 **Confusion Matrix** — for classification tasks
- 🔮 **Live Predictions** — enter values and get instant predictions from any trained model

---

## 🚀 Deploy to Streamlit Community Cloud (Free)

### Step 1 — Push to GitHub

```bash
# Initialize a new repo
git init
git add .
git commit -m "Initial commit: DataML Studio"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/dataml-studio.git
git branch -M main
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **"New app"**
3. Select your repository: `YOUR_USERNAME/dataml-studio`
4. Set **Main file path** to: `app.py`
5. Click **"Deploy!"**

Streamlit Cloud will automatically install `requirements.txt` and launch your app at:
```
https://YOUR_USERNAME-dataml-studio-app-XXXXXX.streamlit.app
```

---

## 💻 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/dataml-studio.git
cd dataml-studio

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📂 File Structure

```
dataml-studio/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🧪 Test Datasets

Try these free CSV datasets:
- [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — Classification
- [Boston Housing](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) — Regression
- [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) — Multi-class Classification
- Any CSV from [Kaggle Datasets](https://www.kaggle.com/datasets)

---

## 🛠 Tech Stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Data | Pandas, NumPy |
| Viz | Plotly |
| ML | Scikit-learn |

---

## 📜 License
MIT — free to use and modify.
