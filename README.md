# Breast Cancer Classification Model

This project is a machine learning model designed to classify breast cancer tumors as either malignant (M) or benign (B) based on a set of features from the Breast Cancer Wisconsin (Diagnostic) dataset. The analysis is conducted in a Jupyter Notebook (`Breast_Cancer_Model.ipynb`).

## What the Project Does

The notebook walks through a complete machine learning workflow:

1.  **Data Loading and Preprocessing**: It loads the `breast-cancer.csv` dataset, cleans it by dropping unnecessary columns, and inspects it for any missing values.

2.  **Exploratory Data Analysis (EDA)**: The project visualizes the class distribution (Malignant vs. Benign) and analyzes the correlation between various features and the final diagnosis to identify the most predictive attributes.

3.  **Feature Engineering**:
    *   The `diagnosis` label is converted from categorical ('M'/'B') to numerical (1/0).
    *   The data is split into an 80% training set and a 20% testing set.
    *   Features are scaled using `StandardScaler` to ensure they have a uniform influence on the model.

4.  **Model Building and Evaluation**:
    *   Several classification algorithms are trained and compared, including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Random Forest, and Gradient Boosting.
    *   The models are evaluated based on accuracy, precision, recall, and F1-score. Confusion matrices are also plotted to compare their performance in distinguishing between classes.

5.  **Hyperparameter Tuning**: `GridSearchCV` is used to fine-tune the hyperparameters of the best-performing model (SVM) to optimize its predictive power.

6.  **Final Model Evaluation**: The optimized SVM is retrained with the best parameters and evaluated using accuracy, a classification report, a confusion matrix, and a ROC curve. The final model achieves an AUC score of **0.997**, demonstrating excellent predictive capability.

## How to Run This Project

This project is a Jupyter Notebook and requires a Python environment with the necessary libraries installed.

### 1. Prerequisites
- Python 3.x
- Jupyter Notebook or JupyterLab. Alternatively, you can upload the notebook to Google Colab.
- The dataset file `breast-cancer.csv` must be located in the same directory as the notebook.

### 2. Install Required Libraries
You can install all necessary libraries using pip. Open your terminal or command prompt and run:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

### 3. Running the Notebook
1.  **Launch Jupyter**: Open your terminal, navigate to the project directory, and start Jupyter Notebook with the command:
    ```bash
    jupyter notebook
    ```
2.  **Open the Notebook**: Your web browser will open the Jupyter interface. Click on `Breast_Cancer_Model.ipynb` to open it.
3.  **Execute the Cells**: You can run the notebook cells sequentially by clicking the "Run" button at the top or by pressing `Shift + Enter` in each cell. This will execute the code and display the outputs, including data tables, visualizations, and model performance metrics.
