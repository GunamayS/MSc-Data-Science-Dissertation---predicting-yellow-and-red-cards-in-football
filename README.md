# 7CCSMPRJ MSc Final Year Project: Predicting Yellow and Red Cards in Football

**Predicting Yellow and Red Cards in Football** is an MSc Final Year Project completed as part of the **MSc Data Science** degree at **King’s College London**.

This project explores the prediction of disciplinary outcomes (yellow and red cards) in professional football matches using statistical and machine-learning models. By analysing historical match data from the English Premier League, the project aims to identify patterns and contributing factors that influence refereeing decisions and player discipline.

The study compares multiple modelling approaches to evaluate their effectiveness in predicting card occurrences, with a particular focus on interpretability, performance, and suitability for count-based football data.

---

## Key Objectives

- Analyse historical football match data related to yellow and red cards  
- Engineer and prepare relevant match-level features  
- Implement and compare multiple predictive models  
- Evaluate model performance using appropriate regression metrics  
- Identify the most suitable modelling approach for disciplinary prediction  

---

## Methods Implemented

- Linear Regression  
- Decision Tree Regression  
- Poisson Regression  
- k-Nearest Neighbours (K-NN)  

Each model is evaluated and compared to assess its strengths, limitations, and predictive accuracy for football disciplinary data.

---

## Dataset

The project uses historical **English Premier League** match data, including match statistics, team information, and disciplinary records.  
The dataset is pre-processed and structured to support both statistical modelling and machine-learning approaches.

---

## Results Summary

Poisson Regression proved to be particularly well-suited for modelling card counts due to the discrete and skewed nature of the target variables.  
Tree-based models captured non-linear relationships, while Linear Regression provided a strong interpretable baseline.  
K-NN performance was sensitive to feature scaling and data sparsity, highlighting its limitations for this task.

---

## Technologies Used

- Python (Jupyter Notebook)  
- pandas, NumPy  
- scikit-learn  
- matplotlib, seaborn  

---

## Requirements

The project uses the Python libraries specified in `requirements.txt`.  
To ensure reproducibility, install the required versions before running the notebooks.

---

## User Guide

### Setting up the Python Libraries

1. Download the `PRJ` folder to a location of your choice  

2. Navigate to the folder location in your terminal  

3. Execute the following command: `pip install -r requirements.txt`
    
### Running the files 
1. In an IDE that supports Jupyter Notebook, open the five files: 
    - `Data Setup.ipynb`
    - `Data Setup Main.py`
    - `Linear Regression.ipynb`
    - `Decision Tree.ipynb`
    - `Poisson Regression.ipynb`
    - `K-NN.ipynb`
2. To execute each file and obtain an output, click the Run All button located at the top of the page 
    - The output tables should be located at the bottom of the code cell
