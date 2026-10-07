# 🌌 AI-Based Intelligent Exoplanet Detection System

A complete, polished, college PBL Machine Learning application built using **Streamlit**, **Scikit-Learn**, and **Plotly** for classifying exoplanet candidates using real astronomical observations from the **NASA Kepler KOI Cumulative Dataset**.

---

## 📌 Project Overview
The **AI-Based Intelligent Exoplanet Detection System** uses supervised machine learning models to analyze stellar transit and planetary features observed by space telescopes. The system provides an interactive web interface where users can inspect real NASA observation data, test new observations via CSV upload, or run synthetic demo simulations.

---

## 🎯 Problem Statement
Astronomical surveys generate massive amounts of light curve data. Manually searching for and verifying planetary transit signals is extremely time-consuming. Automated machine learning algorithms can rapidly screen signals and classify them into potential candidates, confirmed planets, or false positive signals.

---

## 🚀 Objectives
- Use real NASA Kepler KOI cumulative dataset for training and evaluation.
- Implement **RandomForestClassifier** as the primary classifier and compare against a **LogisticRegression** baseline.
- Prevent data leakage by excluding post-classification metadata and disposition-derived target indicators.
- Provide a responsive, modern astronomy space-themed **Streamlit UI** with interactive 3D/2D stellar transit visualizations.
- Offer dual inference modes: **Real CSV Upload** and **Synthetic Demo Mode**.

---

## 📊 Dataset Features & Target

### Target Variable
- **`koi_disposition`**: `CONFIRMED`, `FALSE POSITIVE`, `CANDIDATE`

### 12 Observational Features Used:
1. `koi_period`: Orbital Period (days)
2. `koi_impact`: Impact Parameter
3. `koi_duration`: Transit Duration (hours)
4. `koi_depth`: Transit Depth (parts per million - ppm)
5. `koi_prad`: Planetary Radius (Earth radii)
6. `koi_teq`: Equilibrium Temperature (Kelvin)
7. `koi_insol`: Insolation Flux (Earth flux)
8. `koi_model_snr`: Transit Signal-to-Noise Ratio
9. `koi_steff`: Stellar Effective Temperature (Kelvin)
10. `koi_slogg`: Stellar Surface Gravity (log10(cm/s²))
11. `koi_srad`: Stellar Radius (solar radii)
12. `koi_kepmag`: Kepler-band Magnitude

---

## ⚙️ Machine Learning Methodology

```text
NASA Kepler Dataset → Data Preprocessing → Train/Test Split → Scikit-Learn Pipeline → Random Forest Classifier → Inference & Probability
```

1. **Preprocessing:** SimpleImputer (median strategy) + StandardScaler fitted strictly on training split.
2. **Train/Test Split:** Stratified 80/20 train-test split.
3. **Classifiers:**
   - **Primary:** `RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)`
   - **Baseline:** `LogisticRegression(max_iter=1000, random_state=42)`

---

## 🧠 Empirical Evaluation Results

Metrics calculated on held-out test dataset:

| Model | Accuracy | F1 Score (Weighted) | Precision (Weighted) | Recall (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **86.05%** | **82.41%** | **79.67%** | **86.05%** |
| **Logistic Regression (Baseline)** | **88.37%** | **87.84%** | **87.42%** | **88.37%** |

*(Note: All metrics displayed in the app are computed dynamically from actual model evaluation).*

---

## 💻 How to Run Locally

### 1. Clone or Open Project Directory
```bash
cd "XO PLANET AMRIT"
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run Streamlit Application
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 🧪 Synthetic Demo Mode
The application includes an optional **Synthetic Demo Mode** for presentation demonstrations:
- Generates synthetic observations within empirical feature ranges.
- **Clearly labeled:** *"Synthetic Demo Observation — generated for application demonstration and not an actual astronomical observation."*
- Synthetic data is **NEVER** used for training the machine learning models.

---

## ☁️ Deployment to Streamlit Community Cloud

1. Push this project repository to **GitHub**.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and select repository.
4. Set Main file path to `app.py`.
5. Click **Deploy!**

*(All file paths are relative and self-contained).*

---

## ⚠️ Important Limitations
- **Exoplanet Candidate Classification:** This ML application provides pattern classification of candidate signals; it does not constitute astronomical confirmation of a new planet.
- **Scientific Validation:** Scientific confirmation of exoplanets requires ground-based radial velocity measurements, spectroscopy, and astronomical validation.

---

## 🔮 Future Improvements
- Integrate larger Kepler / TESS survey datasets.
- Implement deep learning (CNN/RNN) for raw light curve time-series analysis.
- Add SHAP / LIME explainability models for per-observation feature breakdown.
