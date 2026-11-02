# Machine-Learning-For-Heart-Failure-Prediction
Application of Machine learning for Predicting the Mortality caused by Heart Failure in Humans

### Executive Summary
Cardiovascular diseases are the disease of the heart and blood vessels. Heart failure (HF) is one of the most prevalent cardiovascular diseases globally and is associated with high morbidity and mortality rates. It is a complex health condition characterized by the heart's inability to pump blood effectively, leading to inadequate blood flow to meet the body's needs.
According to the World Health Organization approximately 26 million people worldwide suffer from heart failure and altogether cardiovascular diseases cause the death of approximately 17.9 million people worldwide annually, contributing approximately 31% to mortality every year.

This study provided a detailed analysis on how a dataset was acquired and preprocessed after which a feature selection approach, feature importance based on Random Forest Classifier was employed for feature engineering. Then, three machine learning algorithms, Logistic Regression, K-Nearest Neighbors, and Random Forest Classifier were utilized for modeling in which the performance of the best classifier was measured in terms of accuracy, recall, precision, f1-score, and AUC value. It turned out that Random Forest Classifier demonstrated the highest accuracy of 87% for predicting HF mortality while Serum Creatinine and Ejection Fraction are indeed the two features that contributed the most in determining the outcome of the model by which heart failure can be predicted. 

 ### Project Overview
 
Being one of the most common cardiovascular diseases, Heart failure (HF) remains one of the leading causes of mortality and morbidity worldwide. The ability to accurately predict mortality in patients with heart failure is critical for improving patient outcomes through timely interventions and personalized treatment strategies. 

This project is focused on building machine learning models capable of predicting the mortality of patients with heart failure and finding the features that can predict their death probability at an early stage.

 ### Data Source

The dataset used in this project was sourced from the Kaggle HF Dataset, which contains 299 entries including 194 males and 105 females between the age of 45 to 95. This dataset comprises of the patient’s clinical records and  diagnostic tests with various attributes like creatine phosphokinase, serum creatinine, ejection fraction, age, diabetes, high BP, platelets, smoking, sex, anemia, and death which is the target variable. 

Creatine phosphokinase (CPK) is an enzyme that can indicate heart disease when it is found in the blood, as it leaks into the bloodstream when there is damage to muscle tissue.  

Serum creatinine is a blood test that measures the level of creatinine in the blood to assess the kidney function. 

Ejection fraction is the amount of blood the heart releases with each contraction. It measures the heart’s pumping efficiency.

Age indicates the length of a patient existence on earth in terms of years. 

Diabetes indicates whether or not the patient has diabetes caused by insufficient insulin production.

High blood pressure represents whether or not the patient has hypertension. 

Platelets is also known as thrombocytes, it depicts blood platelets in the body.

Smoking shows whether the patient is a smoker or not. 

Sex indicates the patient’s gender (female – 0, male – 1). 

Anemia refers to a decrease in red blood cells or hemoglobin. 

Death is the class that classifies whether the patient died – (1) or survived – (0) 


 ### Problem Statement

Heart failure (HF) is a leading cause of morbidity and mortality worldwide, contributing significantly to healthcare burdens. Despite advancements in medical treatments and the development of clinical guidelines, predicting mortality in heart failure patients remains a challenging task. 

Traditional risk assessment tools and statistical models provide some predictive capabilities but are often limited by their reliance on linear assumptions, inability to handle complex and high-dimensional data, and lack of personalization. As heart failure is a multifactorial condition influenced by a wide range of clinical, demographic, and lifestyle factors, there is a need for more sophisticated predictive models that can accurately assess an individual patient’s risk of mortality. 

Machine learning (ML) with its ability to analyze large datasets, model non-linear relationships and integrate diverse data types offers a promising approach to improve mortality prediction in HF patients. 

 ### Objectives

The main objectives of this project are: 
1.	To build a machine learning model capable of predicting the mortality of patients with heart failure. 
2.	To evaluate the performance of the model using evaluation metrics such as accuracy, precision, recall and F1-score. 
3.	To identify key predictive features associated with mortality in heart failure patients. 

 ### Tools and Methodology

  #### Tools Used

- Jupyter Notebook - The Workbench for Code Execution
- Pandas - Data Manipulation and Cleaning
- NumPy - Numerical and Mathematical Computations 
- Matplotlib - Data Visualisation
- Scikit-learn - Machine Learning Model Development

#### Methodology

#### Data Cleaning & Preparation
  
  The following data quality steps were performed to ensure analytical reliability

 1. Checked for Missing and duplicate values to ensure data quality and integrity 
 2. Validation of data types and formats to ensure compatibiity  with analytical operations
   
   
