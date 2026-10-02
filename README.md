# Loan Approval Prediction

A machine learning project that predicts whether a loan application is likely to be approved based on applicant and loan-related information.

The project covers the complete basic machine learning workflow, starting from data cleaning and exploratory data analysis to feature encoding, scaling, model training, evaluation, and feature engineering.

## Project Overview

The dataset contains information about loan applicants such as:

* Applicant income
* Coapplicant income
* Age
* Dependents
* Credit score
* Existing loans
* DTI ratio
* Savings
* Collateral value
* Loan amount
* Loan term
* Employment status
* Marital status
* Loan purpose
* Property area
* Education level
* Gender
* Employer category

The target variable is `Loan_Approved`, which contains two classes:

* `Yes`
* `No`

The dataset contains 1000 records and initially has 20 columns. Around 50 records contain missing values, which are handled during preprocessing.

## Workflow

The project follows these steps:

1. Load the dataset
2. Check the dataset structure
3. Handle missing values
4. Perform exploratory data analysis
5. Remove the applicant ID
6. Encode categorical variables
7. Analyze feature correlations
8. Split the data into training and testing sets
9. Scale the features
10. Train machine learning models
11. Evaluate the models
12. Perform feature engineering
13. Train the models again

## Dataset

The dataset is loaded using pandas:

```python
df = pd.read_csv("loan_approval_data.csv")
```

Basic dataset checks were performed using:

```python
df.head()
df.info()
df.isnull().sum()
df.describe()
```

The dataset contains both numerical and categorical features.

## Data Preprocessing

### Handling Missing Values

Numerical columns are filled using the mean value:

```python
num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])
```

Categorical columns are filled using the most frequent value:

```python
cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])
```

After imputation, there are no missing values in the dataset.

### Removing Applicant ID

`Applicant_ID` was removed because it is an identifier and is not useful as a predictive feature.

```python
df = df.drop("Applicant_ID", axis=1)
```

This leaves 19 columns before encoding.

### Encoding Categorical Features

`Education_Level` and `Loan_Approved` were encoded using `LabelEncoder`.

The remaining categorical features were converted using one-hot encoding with `drop="first"`.

```python
ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)
```

After encoding, the dataset contains 28 columns.

## Exploratory Data Analysis

Several visualizations were used to understand the dataset:

* Loan approval class distribution
* Education level distribution
* Applicant income distribution
* Coapplicant income distribution
* Box plots for numerical variables
* Credit score vs loan approval
* Applicant income vs loan approval
* Correlation heatmap

The target classes contain 722 and 278 records respectively in the analyzed dataset.

### Correlation Analysis

The correlation analysis showed that some of the features had stronger relationships with the target than others.

The two strongest correlations with `Loan_Approved` in the analysis were:

* `Credit_Score`: approximately `0.45`
* `DTI_Ratio`: approximately `-0.44`

`Loan_Amount` also showed a negative correlation of approximately `-0.13`.

Correlation was used as an exploratory analysis and not as the only basis for selecting features.

## Train-Test Split

The data was divided into training and testing sets using an 80/20 split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Standardization was then applied using `StandardScaler`.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The scaler is fitted only on the training data and then used to transform the test data.

## Machine Learning Models

Three classification algorithms were tested:

### 1. Logistic Regression

Logistic Regression was used as one of the baseline classification models.

Initial results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  |  86.5% |
| Precision | 78.33% |
| Recall    | 77.05% |
| F1 Score  | 77.69% |

Confusion matrix:

```text
[[126  13]
 [ 14  47]]
```

### 2. K-Nearest Neighbors

KNN was trained using `n_neighbors=5`.

Initial results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  |  76.0% |
| Precision | 62.75% |
| Recall    | 52.46% |
| F1 Score  | 57.14% |

Confusion matrix:

```text
[[120  19]
 [ 29  32]]
```

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes was also tested.

Initial results:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  |  86.5% |
| Precision | 80.36% |
| Recall    | 73.77% |
| F1 Score  | 76.92% |

Confusion matrix:

```text
[[128  11]
 [ 16  45]]
```

Based on the initial precision comparison in the notebook, Naive Bayes was selected for precision.

## Feature Engineering

Additional features were created from `DTI_Ratio` and `Credit_Score`:

```python
df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2
```

The original `Credit_Score` and `DTI_Ratio` columns were then removed from the model input while their squared versions were retained.

```python
X = df.drop(
    columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"]
)
```

The data was then split and scaled again before retraining the models.

## Results After Feature Engineering

After feature engineering, the Logistic Regression model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  |  88.0% |
| Precision | 78.46% |
| Recall    | 83.61% |
| F1 Score  | 80.95% |

Confusion matrix:

```text
[[125  14]
 [ 10  51]]
```

The KNN model achieved:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  |  78.5% |
| Precision | 67.31% |
| Recall    | 57.38% |
| F1 Score  | 61.95% |

The notebook also retrains Gaussian Naive Bayes after feature engineering, but the provided content does not show its final printed metric values, so no additional result is claimed here.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Machine Learning Concepts Used

* Data preprocessing
* Missing value imputation
* Exploratory Data Analysis
* Label Encoding
* One-Hot Encoding
* Correlation analysis
* Train-test split
* Feature scaling
* Logistic Regression
* K-Nearest Neighbors
* Gaussian Naive Bayes
* Feature engineering
* Classification metrics
* Confusion matrix

## Project Structure

```text
Loan-Approval-Prediction/
│
├── loan_approval_data.csv
├── loan_approval_prediction.ipynb
└── README.md
```

## How to Run

Clone the repository and install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Open the notebook:

```bash
jupyter notebook
```

Then run the cells in `loan_approval_prediction.ipynb`.

Make sure `loan_approval_data.csv` is available in the same project directory as the notebook.

## Conclusion

This project demonstrates a complete machine learning workflow for a binary loan approval classification problem.

Different classification algorithms were tested and compared using accuracy, precision, recall, F1 score, and confusion matrices. Feature engineering was also performed to check whether transformed features could improve model performance.

The results show that Logistic Regression improved from 86.5% to 88.0% accuracy after the feature engineering step used in the project.
