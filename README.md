# Sampling Techniques for Imbalanced Data - ML Model Comparison

## 📌 Overview
This project demonstrates **5 different sampling techniques** to handle imbalanced datasets and compares the performance of **5 machine learning models** on each sampling method. The dataset used is a credit card fraud detection dataset.

## 📊 Dataset
- **File:** `Creditcard_data.csv`
- **Target Variable:** `Class` (0 = Non-Fraud, 1 = Fraud)
- **Problem:** Highly imbalanced dataset with very few fraud cases

## 🔄 Sampling Techniques Used

| # | Technique | Type | Description |
|---|-----------|------|-------------|
| 1 | **Random Under-Sampling** | Under-sampling | Randomly removes majority class samples |
| 2 | **Random Over-Sampling** | Over-sampling | Randomly duplicates minority class samples |
| 3 | **Tomek Links** | Under-sampling | Removes majority samples that form Tomek links with minority samples |
| 4 | **SMOTE** | Over-sampling | Generates synthetic minority samples using K-Nearest Neighbors |
| 5 | **NearMiss** | Under-sampling | Selects majority samples closest to minority samples |

## 🤖 Machine Learning Models

1. **Logistic Regression** - Linear classifier for binary classification
2. **Decision Tree** - Tree-based non-linear classifier
3. **Random Forest** - Ensemble of decision trees
4. **Support Vector Machine (SVM)** - Hyperplane-based classifier
5. **K-Nearest Neighbors (KNN)** - Distance-based classifier

## 📈 Results - Accuracy Comparison

| Model | Random Under-Sampling | Random Over-Sampling | Tomek Links | SMOTE | NearMiss |
|-------|----------------------|---------------------|-------------|-------|----------|
| Decision Tree | 0.5000 | 0.9891 | 0.9826 | 0.9672 | 0.1667 |
| KNN | 0.1667 | 0.9847 | 0.9870 | 0.8362 | 0.8333 |
| Logistic Regression | 0.8333 | 0.9192 | 0.9870 | 0.9258 | 0.5000 |
| Random Forest | 0.6667 | **1.0000** | 0.9870 | 0.9934 | 0.1667 |
| SVM | 0.1667 | 0.6856 | 0.9870 | 0.6900 | 0.1667 |

### 🏆 Key Findings

- **Best Overall:** Random Forest with Random Over-Sampling achieves **100% accuracy**
- **Most Consistent:** Tomek Links provides stable performance (~98.7%) across most models
- **SMOTE Performance:** Works well with tree-based models (Decision Tree, Random Forest)
- **Under-sampling Limitation:** Random Under-Sampling and NearMiss show poor performance due to information loss

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### Dependencies
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `scikit-learn` - Machine learning models and metrics
- `imbalanced-learn` - Sampling techniques (SMOTE, Tomek Links, etc.)

## 📁 Project Structure

```
├── Sampling_Assignment.ipynb    # Main Jupyter notebook with code
├── Creditcard_data.csv          # Input dataset
├── accuracy_comparison.csv      # Results pivot table
├── output.csv                   # Detailed results with F1-scores
└── README.md                    # Project documentation
```

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the Jupyter notebook
   ```bash
   jupyter notebook Sampling_Assignment.ipynb
   ```

## 📝 Conclusion

| Sampling Method | Best Model | Accuracy |
|-----------------|------------|----------|
| Random Under-Sampling | Logistic Regression | 83.33% |
| Random Over-Sampling | Random Forest | 100.00% |
| Tomek Links | KNN / Logistic Regression / Random Forest | 98.70% |
| SMOTE | Random Forest | 99.34% |
| NearMiss | KNN | 83.33% |

**Recommendation:** For this credit card fraud dataset, **Random Over-Sampling with Random Forest** or **SMOTE with Random Forest** provides the best results.

## 📚 References

- [imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 👤 Author
Rehnoor Aulakh <br>
Thapar University <br>
Predictive Analytics Assignment - Sampling Techniques

---
⭐ If you found this helpful, please star the repository!
