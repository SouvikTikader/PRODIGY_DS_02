
````markdown
# 🚢 PRODIGY_DS_02 – Titanic Survival Data Analysis & Visualization

This project performs **exploratory data analysis (EDA)** on the Titanic dataset using Python. It visualizes survival trends based on features like **age, sex, class, and fare**, and also includes correlation heatmaps and pairplots to explore feature relationships.

---

## 📁 Dataset

- `train.csv` – contains training data with survival labels
- `test.csv` – contains test data without labels  
- Source: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic/data)

---

## 📊 Visualizations

Saved under the `images/` directory:

| Plot | Description |
|------|-------------|
| `survival_count.png` | Count of survivors and non-survivors |
| `survival_by_sex.png` | Survival rate grouped by gender |
| `survival_by_pclass.png` | Survival rate grouped by passenger class |
| `age_distribution_step_count.png` | Age distribution (count-based histogram) by survival |
| `fare_distribution_by_survival.png` | Fare distribution (count-based histogram) by survival |
| `correlation_heatmap.png` | Correlation heatmap of numeric features |
| `pairplot_selected.png` | Pairplot of selected features colored by survival status |

---

## 📂 Output Files

- `outputs/cleaned_train_data.csv` – Cleaned and processed training dataset  
- `outputs/cleaned_test_data.csv` – Cleaned test dataset (ready for prediction)  

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SouvikTikader/PRODIGY_DS_02.git
   cd PRODIGY_DS_02
````

2. **Install Required Packages**

   ```bash
   pip install pandas seaborn matplotlib
   ```

3. **Run the Script**

   ```bash
   python titanic_eda.py
   ```

The script will:

* Clean and preprocess the dataset
* Generate and save insightful visualizations
* Output cleaned `.csv` files for further modeling

---

## 📧 Author

**Souvik Tikader**
GitHub: [SouvikTikader](https://github.com/SouvikTikader)

---
