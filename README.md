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

The figure 1.1 below shows the information of the dataset after cleaning. It clearly indicated that there are no missing values in all the 299 entries for each column.



<img width="528" height="295" alt="dataset info" src="https://github.com/user-attachments/assets/4da9e640-2624-42c8-afd0-57f3e5195199" />


#### Exploratory Data Analysis (EDA)

Correlation matrix for EDA: A correlation matrix showing the correlation coefficients between variables was conducted. This helps to identify relationships between variables such as positive and negative correlation and also measures the strength of the relationship ranging from -1 (perfect negative) to 1 (perfect positive) (Vigneshwer Dhinakaran, 2018).

Figure 1.2 represents a correlation matrix that explores the relationship between the various variables in the dataset. Serum creatinine shows the greatest positive strength having 0.29 while ejection fraction shows the greatest negative strength having -0.27.

<img width="692" height="542" alt="Correlation Matrix" src="https://github.com/user-attachments/assets/fdaedacd-1563-480f-a656-cf331b87daf5" />


#### Feature Selection

Feature Selection is the process of selecting relevant features (variables) from the original dataset for model training. Relevant features were selected based on clinical significance, correlation analysis, and feature importance from preliminary models. This step helped reduce the dimensionality of the dataset and improves model performance.

Figure 1.3 shows the feature importance of all the variables in the dataset. Features with importance less than 0.05 were dropped why features with importance greater than 0.05 were used for modelling.

<img width="1159" height="614" alt="Feature importance" src="https://github.com/user-attachments/assets/461c6e8a-5b6f-48ba-9483-35b08d6a00fe" />


#### Balancing of Imbalanced Dataset
An imbalanced dataset is a dataset where the classes in the target variable are not equally represented. One class has a significantly larger number of instances than the other. This imbalance can make the machine learning models to be baised towards the majority class, neglecting the minority class 

The imbalanced dataset was rendered balanced using the SMOTE method (Synthetic Minority Over-sampling Technique) to prevent the models from being biased in their predictions and accuracies.  

Figure 1.4 shows the bar chart distribution of the two classes in the target variable. The label “0” represents “no death by HF” while the label “1” represents “death by HF”. The ratio of “no death by HF” to “death by HF” is     203 : 96. This represent an imbalanced dataset.


<img width="768" height="563" alt="target variable" src="https://github.com/user-attachments/assets/355c67eb-5612-4837-88dd-02d8496615d3" />
                                               
Figure 1.5 shows the bar chart distribution of the two classes in the target variable after the dataset was rendered balanced using the SMOTE method.

 <img width="782" height="565" alt="BALANCED DATASET" src="https://github.com/user-attachments/assets/e18b8a11-3da6-47b5-8ab6-dae7bdbd7a93" />


#### Data Splitting

Data splitting in modeling is the process of dividing the dataset into two or more subsets basically for training,
testing and or validation. The dataset was split into training and testing sets (80% for training and 20% for 
testing)  

 #### Model Selection
The following machine learning algorithms were selected for modeling: 

#### Random Forests Classifier. 
Random Forest Classifier is a trending machine learning technique which is capable of both regression and 
classiﬁcation. It is a supervised learning algorithm in which the ground methodology is recursion. In this 
algorithm, a group of decision trees are being created and the bagging method is used for training purposes. 
Random Forest Classifier is insensitive to noise and can be used for imbalanced datasets.

#### Logistic Regression
Logistic Regression (LR) is a popular mathematical modelling procedure which is used for epidemiologic
datasets in the area of machine learning. It ﬁrst calculates using the logistic function then it learns the 
coeﬃcients for the logistic regression model and then ﬁnally makes predictions using that logistic regression 
model. This model is a generalized linear model and has two parts, namely, linear part and link function. The 
linear part is responsible for carrying out the calculations of the classiﬁcation model, and the link function is 
responsible for delivering the output of the calculation. LR is a supervised machine learning algorithm
which needs a hypothesis and a cost function. It is to be noted that optimizing the cost function is important.

#### K-Nearest Neighbors (KNN)
K-Nearest Neighbors (KNN) is an “instance-based learning” or non-generalizing learning algorithm. It does not
focus on constructing a general internal model; instead, it stores all instances corresponding to training data in 
n-dimensional space. KNN uses data and classifies new data points based on similarity measures (e.g., 
Euclidean distance function). Classification is computed from a simple majority vote of the k nearest neighbors 
of each point. It is quite robust to noisy training data, and accuracy depends on the data quality. The biggest 
issue with KNN is to choose the optimal number of neighbors to be considered. KNN can be used both for 
classification as well as regression.

#### Model Training
The selected models were trained on the training set and tested on the testing set. The training process involves feeding the models with input features and corresponding labels (i.e., whether the patient survived or not) to learn the underlying patterns and relationships 

#### Hyperparameter Tuning 
Hyperparameter tuning is the process of finding the optimal combination of hyperparameters to train a machine 
learning model. It helps to improve the model performance and avoid overfitting or underfitting. 
Hyperparameter tuning tools like GridSearchCv and RandomizedSearchCV was used to tuned the baseline 
models to improve their predicting accuracies.

### Result and Discussion 

#### Model Accuracy
After the training and the testing of the models on the dataset, a baseline model accuracy was achieved for each 
model. Logistic regression, KNN and random forest classification demonstrated a model accuracy of 0.78, 0.59, 0.74 respectively.
Figure 1.6 is the representation of the baseline model accuracy scores 

<img width="835" height="763" alt="new accuracy score" src="https://github.com/user-attachments/assets/e94d2a72-49d7-4900-ae6a-1a62fb919fee" />

#### Hyperparameter Tuning
Logistic Regression decreased in its accuracy for prediction from 0.78 to 0.77.
KNN increased in its accuracy for prediction from 0.59 to 0.69 and 
Random Forest Classifier increased in its accuracy for prediction from 0.84 to 0.87.

Random Forest Classifier demonstrated the highest accuracy of 0.87 and was picked to be the representation model.

MODELS	                     BEFORE TUNING	   AFTER TUNING
LOGISTIC REGRESSION 	            0.78	           0.77
KNN	                             0.59	           0.69
RANDOM FOREST CLASSIFIER	        0.84	           0.87

#### Evaluation Metrics
The performance of the best model for predicting HF mortality was evaluated using the following evaluation 
metrics:
Accuracy: This measures the overall correctness of the model but can be misleading in imbalanced datasets. 

Precision: Precision measures the proportion of true positive among all positive predictions. 

Recall: Recall measures the proportion of true positive among all actual positive instances. 

F1-Score: The harmonic mean of precision and recall, providing a single metric that balances both.

AUC-ROC Curve: The area under the receiver operating characteristic curve (AUC-ROC) is used to assess the model's ability to distinguish between classes (e.g., mortality vs. survival). 
  

 
