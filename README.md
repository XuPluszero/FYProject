# FYProject - IDH Mutation Prediction in Brain Tumors

## 🎯 Project Overview

This is a **Final Year Project** focused on **IDH mutation prediction in brain tumors** using **radiomics analysis** and **machine learning ensemble methods**. The project analyzes medical imaging data from three MRI modalities (ADC, T1C, T2) to predict IDH (Isocitrate Dehydrogenase) mutation status, which is crucial for brain tumor diagnosis and treatment planning.

### Key Features
- **Multi-modal radiomics analysis** using PyRadiomics-extracted features
- **Advanced feature selection** with Lasso regularization
- **Multiple machine learning classifiers** comparison
- **Ensemble learning strategies** (Averaging, Stacking, Voting)
- **Comprehensive performance evaluation** with ROC analysis

## 🛠️ Technology Stack

### Core Libraries
- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and preprocessing
- **XGBoost** - Gradient boosting framework
- **LightGBM** - Gradient boosting framework
- **Matplotlib** - Data visualization and plotting

### Machine Learning Components
- **Feature Selection**: Lasso Regression with Cross-Validation
- **Classifiers**: 
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
  - LightGBM
- **Ensemble Methods**:
  - Probability Averaging
  - Stacking (Meta-classifier)
  - Majority Voting
- **Evaluation**: Stratified K-Fold Cross-Validation, ROC-AUC Analysis

## 📋 Requirements

Create a `requirements.txt` file with the following dependencies:

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
pyradiomics>=3.1.0
```

## 🚀 Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/XuPluszero/FYProject.git
cd FYProject
```

2. **Create virtual environment** (Recommended)
```bash
python -m venv fyp_env
source fyp_env/bin/activate  # On Windows: fyp_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare data**
   - Ensure all CSV files are in the `data/` directory
   - The project expects the following files:
     - `info_tumor.csv` - Patient labels with IDH mutation status ✅ (included)
     - Feature files for each modality (ADC, T1C, T2) for both edema and tumor regions
   - **Note**: Large feature CSV files (>100MB) are not included in this repository due to GitHub size limits. Contact the author for access to the complete dataset.

## 📊 Usage

### Basic Execution
```bash
python fyp.py
```

### What the Script Does:

1. **Data Loading & Preprocessing**
   - Loads radiomics features from CSV files
   - Merges edema and tumor features for each MRI modality
   - Handles missing values and standardizes features

2. **Feature Selection**
   - Applies Lasso regularization with cross-validation
   - Selects most relevant features for each modality

3. **Model Training & Evaluation**
   - Trains 5 different classifiers on each modality
   - Uses stratified 5-fold cross-validation
   - Generates ROC curves and calculates AUC scores

4. **Ensemble Learning**
   - Combines predictions from all three modalities
   - Implements three ensemble strategies
   - Compares ensemble vs single-modality performance

### Expected Output:
- Feature selection results for each modality
- Individual classifier performance metrics
- ROC curve plots for each modality
- Ensemble method comparison
- Final performance comparison plot

## 📈 Results

The project evaluates:
- **Individual Modality Performance**: ADC, T1C, T2
- **Classifier Comparison**: LR, RF, SVM, XGBoost, LightGBM
- **Ensemble Strategies**: Averaging, Stacking, Voting
- **Performance Metrics**: AUC-ROC, Accuracy

## 📁 Project Structure

```
FYProject/
├── fyp.py                 # Main analysis script
├── data/                  # Data directory
│   ├── info_tumor.csv     # Patient labels
│   └── features_*.csv     # Radiomics features
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── Final_Report_F.pdf     # Detailed project report
├── user manual.pdf        # User guide
└── FYP Pre.pptx          # Project presentation
```

## 📚 Documentation

- **📄 Final Report**: `Final_Report_F.pdf` - Comprehensive technical report
- **📖 User Manual**: `user manual.pdf` - Step-by-step usage guide
- **🎯 Presentation**: `FYP Pre.pptx` - Project overview slides

## 🎓 Academic Context

This project was developed as part of **SEEM4999 (Final Year Project)** at **The Chinese University of Hong Kong**. It demonstrates the application of machine learning in medical imaging analysis, specifically for brain tumor IDH mutation prediction.

### Key Learning Outcomes:
- Medical imaging data analysis
- Radiomics feature extraction and selection
- Multi-modal machine learning
- Ensemble learning techniques
- Cross-validation and model evaluation

## 📄 License

This project is part of academic work for SEEM4999. All rights reserved.

---

*This project showcases proficiency in machine learning, data science, and medical imaging analysis - suitable for demonstrating technical skills to potential employers.* 
