# Machine-Learning-For-Heart-Failure-Prediction
## Application of Machine learning for Predicting the Mortality caused by Heart Failure in Humans

### Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Machine Learning in Healthcare](#machine-learning-in-healthcare)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Data Source](#data-source)
- [Tools and Methodology](#tools-and-methodology)
- [Result and Discussion](#result-and-discussion)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [References](#references)

### Abstract

Being one of the most common cardiovascular diseases, Heart failure (HF) remains one of the leading causes of mortality and morbidity worldwide. The ability to accurately predict mortality in patients with heart failure is critical for improving patient outcomes through timely interventions and personalized treatment strategies. This thesis is focused on building machine learning models capable of predicting the mortality of patients with heart failure and finding the features that can predict their death probability at an early stage. 

This study provided a detailed analysis on how a dataset was acquired and preprocessed after which a feature selection approach, feature importance based on Random Forest Classifier was employed for feature engineering. Then, three machine learning algorithms, Logistic Regression, K-Nearest Neighbors, and Random Forest Classifier were utilized for modeling in which the performance of the best classifier was measured in terms of accuracy, recall, precision, f1-score, and AUC value. It turned out that Random Forest Classifier demonstrated the highest accuracy of 87% for predicting HF mortality while Serum Creatinine and Ejection Fraction are indeed the two features that contributed the most in determining the outcome of the model by which heart failure can be predicted. 
 

 ### Introduction
 
Cardiovascular diseases are the disease of the heart and blood vessels. Heart failure (HF) is one of the most prevalent cardiovascular diseases globally and is associated with high morbidity and mortality rates. It is a complex health condition characterized by the heart's inability to pump blood effectively, leading to inadequate blood flow to meet the body's needs. 

According to the World Health Organization approximately 26 million people worldwide suffer from heart failure and altogether cardiovascular diseases cause the death of approximately 17.9 million people worldwide annually, contributing approximately 31% to mortality every year (Kedia & Bhushan, 2022)  (Chicco & Jurman, 2020)

Heart failure can result from various underlying health conditions, including coronary artery disease, high blood pressure, diabetes, obesity and cardiomyopathy. The condition is typically classified based on the ejection fraction, a measure of the heart's pumping ability, into heart failure with reduced ejection fraction (HFrEF) and heart failure with preserved ejection fraction (HFpEF) (Park et al., 2018). Despite advancements in treatment, heart failure remains a leading cause of hospitalization and death, particularly among older adults. 

 ### Machine Learning in Healthcare 

The term ML was coined in 1959 by Arthur Samuel and refers to the idea that computer algorithms will learn and adjust and thereby improve automatically within their given boundaries through the use of data (Kresoja et al., 2023).

Machine learning is a subset of artificial intelligence and has gained significant attention in healthcare for its ability to analyze complex datasets and identify patterns that traditional statistical methods might miss. Machine learning models can process large amounts of data, identify non-linear relationships, and improve predictive accuracy in various clinical applications, including disease diagnosis, treatment planning, and medical prognosis.

 ### Problem Statement

Heart failure (HF) is a leading cause of morbidity and mortality worldwide, contributing significantly to healthcare burdens. Despite advancements in medical treatments and the development of clinical guidelines, predicting mortality in heart failure patients remains a challenging task. Traditional risk assessment tools and statistical models provide some predictive capabilities but are often limited by their reliance on linear assumptions, inability to handle complex and high-dimensional data, and lack of personalization. 

As heart failure is a multifactorial condition influenced by a wide range of clinical, demographic, and lifestyle factors, there is a need for more sophisticated predictive models that can accurately assess an individual patient’s risk of mortality. Machine learning (ML), with its ability to analyze large datasets, model non-linear relationships, and integrate diverse data types, offers a promising approach to improve mortality prediction in HF patients. 

 ### Objectives
The main objectives of this project are: 
1.	To build a machine learning model capable of predicting the mortality of patients with heart failure. 
2.	To evaluate the performance of the model using evaluation metrics such as accuracy, precision, recall and F1-score. 
3.	To identify key predictive features associated with mortality in heart failure patients. 

 ### Data Source

The dataset used in this project was sourced from the Kaggle HF Dataset, which contains 299 entries including 194 males and 105 females between the age of 45 to 95. This dataset comprises of the patient’s clinical records and  diagnostic tests with various attributes like creatine phosphokinase, serum creatinine, ejection fraction, age, diabetes, high BP, platelets, smoking, sex, anemia, and death which is the target variable. 

- Creatine phosphokinase (CPK) is an enzyme that can indicate heart disease when it is found in the blood, as it leaks into the bloodstream when there is damage to muscle tissue.  

- Serum creatinine is a blood test that measures the level of creatinine in the blood to assess the kidney function. 

- Ejection fraction is the amount of blood the heart releases with each contraction. It measures the heart’s pumping efficiency.

- Age indicates the length of a patient existence on earth in terms of years. 

- Diabetes indicates whether or not the patient has diabetes caused by insufficient insulin production.

- High blood pressure represents whether or not the patient has hypertension. 

- Platelets is also known as thrombocytes, it depicts blood platelets in the body.

- Smoking shows whether the patient is a smoker or not. 

- Sex indicates the patient’s gender (female – 0, male – 1). 

- Anemia refers to a decrease in red blood cells or hemoglobin. 

- Death is the class that classifies whether the patient died – (1) or survived – (0) 

 ### Tools and Methodology

  #### Tools Used

- Jupyter Notebook - The Workbench for Code Execution
- Pandas - Data Manipulation and Cleaning
- NumPy - Numerical and Mathematical Computations 
- Matplotlib - Data Visualisation
- Scikit-learn - Machine Learning and Model Development

#### Methodology

#### Data Cleaning & Preparation
  
 The following data quality steps were performed to ensure analytical reliability

 1. Checked for Missing and duplicate values to ensure data quality and integrity 
 2. Validation of data types and formats to ensure compatibiity  with analytical operations

The figure 1 below shows the information of the dataset after cleaning. It clearly indicated that there are no missing values in all the 299 entries for each column.

Figure 1: Dataset Information

<img width="528" height="295" alt="dataset info" src="https://github.com/user-attachments/assets/4da9e640-2624-42c8-afd0-57f3e5195199" />


#### Exploratory Data Analysis (EDA)

Correlation matrix for EDA: A correlation matrix showing the correlation coefficients between variables was conducted. This helps to identify relationships between variables such as positive and negative correlation and also measures the strength of the relationship ranging from -1 (perfect negative) to 1 (perfect positive) (Vigneshwer Dhinakaran, 2018).

Figure 1.2 represents a correlation matrix that explores the relationship between the various variables in the dataset. Serum creatinine shows the greatest positive strength having 0.29 while ejection fraction shows the greatest negative strength having -0.27.

Figure 2: Confusion Matrix
<img width="692" height="542" alt="Correlation Matrix" src="https://github.com/user-attachments/assets/fdaedacd-1563-480f-a656-cf331b87daf5" />


#### Feature Selection

Feature Selection is the process of selecting relevant features (variables) from the original dataset for model training. Relevant features were selected based on clinical significance, correlation analysis, and feature importance from preliminary models. This step helped reduce the dimensionality of the dataset and improves model performance.

Figure 3 shows the feature importance of all the variables in the dataset. Features with importance less than 0.05 were dropped why features with importance greater than 0.05 were used for modelling.

Figure 3: Feature Importance
<img width="1159" height="614" alt="Feature importance" src="https://github.com/user-attachments/assets/461c6e8a-5b6f-48ba-9483-35b08d6a00fe" />


#### Balancing of Imbalanced Dataset
An imbalanced dataset is a dataset where the classes in the target variable are not equally represented. One class has a significantly larger number of instances than the other. This imbalance can make the machine learning models to be baised towards the majority class, neglecting the minority class 

The imbalanced dataset was rendered balanced using the SMOTE method (Synthetic Minority Over-sampling Technique) to prevent the models from being biased in their predictions and accuracies.  

Figure 4 shows the bar chart distribution of the two classes in the target variable. The label “0” represents “no death by HF” while the label “1” represents “death by HF”. The ratio of “no death by HF” to “death by HF” is     203 : 96. This represent an imbalanced dataset.

Figure 4: Target Variable Distribution
<img width="768" height="563" alt="target variable" src="https://github.com/user-attachments/assets/355c67eb-5612-4837-88dd-02d8496615d3" />
                                               
Figure 5 shows the bar chart distribution of the two classes in the target variable after the dataset was rendered balanced using the SMOTE method.

Figure 5: Balanced Dataset
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
Figure 6 is the representation of the baseline model accuracy scores 

Figure 6: Baseline Model Scores
<img width="835" height="763" alt="new accuracy score" src="https://github.com/user-attachments/assets/e94d2a72-49d7-4900-ae6a-1a62fb919fee" />

#### Hyperparameter Tuning

- Logistic Regression decreased in its accuracy for prediction from 0.78 to 0.77.
- KNN increased in its accuracy for prediction from 0.59 to 0.69 and
- Random Forest Classifier increased in its accuracy for prediction from 0.84 to 0.87.

Random Forest Classifier demonstrated the highest accuracy of 0.87 and was picked to be the representation model.

Figure 7: Result of Hyperparameter Tuning
<img width="976" height="229" alt="Result of Hyperparameter Tuning " src="https://github.com/user-attachments/assets/3eb75613-f0cb-4e64-bc2c-74b68c89cc81" />


#### Evaluation Metrics
The performance of the best model for predicting HF mortality was evaluated using the following evaluation 
metrics:

Accuracy: This measures the overall correctness of the model but can be misleading in imbalanced datasets. 

Precision: Precision measures the proportion of true positive among all positive predictions. 

Recall: Recall measures the proportion of true positive among all actual positive instances. 

F1-Score: The harmonic mean of precision and recall, providing a single metric that balances both.

Figure 8: Evaluation Metrics
<img width="1003" height="318" alt="Evaluation metrics" src="https://github.com/user-attachments/assets/66cfe55e-4021-48b5-b5d2-0ff2c7ad8dd1" />


AUC-ROC Curve: The area under the receiver operating characteristic curve (AUC-ROC) is used to assess the model's ability to distinguish between classes (e.g., mortality vs. survival). 

Figure 9: AOC-ROC Curve of the Best Model
<img width="746" height="693" alt="AOC NEW" src="https://github.com/user-attachments/assets/f9f5ba72-f446-429c-ad90-847f350e09e6" />
 
#### Confusion Matrix

A confusion matrix which summarizes the performance of a machine learning model on a set of test data was comdcuted. It is a means of displaying the number of accurate and inaccurate instances based on the model’s predictions. It is often used to measure the performance of classification models, which aim to predict a categorical label for each input instance.
The matrix displays the number of instances produced by the model on the test data. These instances are

- True Positive (TP): The model correctly predicted a positive outcome (the actual outcome was positive).
- True Negative (TN): The model correctly predicted a negative outcome (the actual outcome was negative).
- False Positive (FP):The model incorrectly predicted a positive outcome (the actual outcome was negative). Also known as a Type I error.
- False Negative (FN): The model incorrectly predicted a negative outcome (the actual outcome was positive). Also known as a Type II error (GeeksforGeeks, 2024).

The basic structure of a confusion matrix is well represented in figure 10

Figure 10: Basic Structure of a confusion Matrix
<img width="651" height="544" alt="basic structure of confusion matrix" src="https://github.com/user-attachments/assets/aba505c1-be6a-4618-a84e-d88a3ebbec05" />

The confusion matrix of the best model for predicting HF mortality is represented in figure 11

Figure 11: Confusion Matrix of the Best Model

<img width="456" height="401" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/238488b4-a360-4b3b-b287-08c04311c40d" />

#### Discussions
The Random Forest Classifier demonstrated the highest accuracy of 87%, precision of 89%, recall of 88%, f1 score of 87% and AUC-ROC curve of 92% for predicting HF mortality while KNN Classifiers demonstrated the lowest accuracy (69%) for predicting HF mortality. It was also observed that Serum Creatinine and Ejection Fraction contributed most in the determining the outcome of the model. 
 
 ### Limitations

While this research project provides meaningful insights, the following limitations must be acknowledged to ensure realistic interpretation of findings:

1. Data Imbalance: An imbalanced dataset is a dataset where the classes in the target variable are not equally represented. One class has a significantly larger number of instances than the other. This imbalance can make the machine learning models to be baised towards the majority class, neglecting the minority class (Ydata, 2023). 
The imbalanced dataset was rendered balanced using the SMOTE method (Synthetic Minority Over-sampling Technique) to prevent the models from being biased in their predictions and accuracies.  

2. Incomplete and Missing Data: Clinical datasets often contain missing values due to incomplete patient records, non-standardized data collection processes, and the loss of follow-up. Missing data can lead to biased predictions and reduce the accuracy of machine learning models.

3. Ethical and Legal Considerations: The use of machine learning in healthcare raises important ethical questions, such as data privacy, potential biases in model predictions, and the implications of using automated decision-making tools in patient care. 
                                 
 ### Conclusion  

This research study resulted in the development of a machine learning model capable of predicting the mortality of heart failure in humans. The model developed achieved an accuracy, precision, recall and f1 score of 87%, 89%, 88% and 87% respectively for predicting HF mortality. This demonstrated the practical application of machine learning techniques in extracting meaningful information from data and developing predictive models.                                                 
Ultimately, this project reinforces the importance of data science and machine learning as valuable tools in adaddressing contemporary healthcare challenges. Future  improvements could explore advnce feature engineering, hyparameter optimization, ensemble learning, cross validation, and external model validation to enhance predictive performance and genralisability.
 
 ### References
 
Al Mehedi Hasan, M., Shin, J., Das, U., & Yakin Srizon, A. (2021). Identifying Prognostic Features for Predicting Heart Failure by Using Machine Learning Algorithm. ACM International Conference Proceeding Series, 40–46. https://doi.org/10.1145/3460238.3460245

Alexander S. Grills. (n.d.). Data Spliting.

Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making, 20(1), 1–16. https://doi.org/10.1186/S12911-020-1023-5/TABLES/11

Choi, E., Schuetz, A., Stewart, W. F., & Sun, J. (2017). Using recurrent neural network models for early detection of heart failure onset. Journal of the American Medical Informatics Association, 24(2), 361–370. https://doi.org/10.1093/JAMIA/OCW112

Deo, R. C. (2015). Machine learning in medicine. Circulation, 132(20), 1920–1930. https://doi.org/10.1161/CIRCULATIONAHA.115.001593/ASSET/F82A0780-3EFF-4CE8-839D-5FF553B00121/ASSETS/GRAPHIC/1920FIG04.JPEG
Freecoursesite. (n.d.). Complete Machine Learning and Data Science Zero to Mastery.

GeeksforGeeks. (2024, June). confusion matrix.

Izraiq, M., Alawaisheh, R., Hamam, I., Hajjiri, M., Jarrad, I., Albustanji, Q., Ahmed, Y., Abu-Dhaim, O., Zuraik, I., Toubasi, A., Dmour, M., & Abu-Hantash, H. (2024). Machine Learning-Driven Mortality Prediction in Heart Failure Patients with Atrial Fibrillation: Evidence from the Jordanian Heart Failure Registry. Research Reports in Clinical Cardiology, Volume 15, 35–44. https://doi.org/10.2147/RRCC.S464317

Jason Brownie. (2020, July 20). Feature selection.

Jayatilake, S. M. D. A. C., & Ganegoda, G. U. (2021). Involvement of Machine Learning Tools in Healthcare Decision Making. Journal of Healthcare Engineering, 2021(1), 6679512. https://doi.org/10.1155/2021/6679512

Kedia, S., & Bhushan, M. (2022). Prediction of Mortality from Heart Failure using Machine Learning. 2022 2nd International Conference on Emerging Frontiers in Electrical and Electronic Technologies, ICEFEET 2022. https://doi.org/10.1109/ICEFEET51821.2022.9848348

Kresoja, K. P., Unterhuber, M., Wachter, R., Thiele, H., & Lurz, P. (2023). A cardiologist’s guide to machine learning in cardiovascular disease prognosis prediction. Basic Research in Cardiology, 118(1), 1–12. https://doi.org/10.1007/S00395-023-00982-7/TABLES/2

Levy, W. C., Mozaffarian, D., Linker, D. T., Sutradhar, S. C., Anker, S. D., Cropp, A. B., Anand, I., Maggioni, A., Burton, P., Sullivan, M. D., Pitt, B., Poole-Wilson, P. A., Mann, D. L., & Packer, M. (2006). The Seattle Heart Failure Model: Prediction of survival in heart failure. Circulation, 113(11), 1424–1433. https://doi.org/10.1161/CIRCULATIONAHA.105.584102

Nisha Chauhan. (2020, October). EDA.

Oladipupo, T. (2010). Types of Machine Learning Algorithms. New Advances in Machine Learning. https://doi.org/10.5772/9385

Park, J. J., Park, J. B., Park, J. H., & Cho, G. Y. (2018). Global Longitudinal Strain to Predict Mortality in Patients With Acute Heart Failure. Journal of the American College of Cardiology, 71(18), 1947–1957. https://doi.org/10.1016/J.JACC.2018.02.064

Sarker, I. H. (2021). Machine Learning: Algorithms, Real-World Applications and Research Directions. SN Computer Science, 2(3), 1–21. https://doi.org/10.1007/S42979-021-00592-X/FIGURES/11

Souza, V. S., & Lima, D. A. (2024). Identifying Risk Factors for Heart Failure: A Case Study Employing Data Mining Algorithms. Journal of Data Science and Intelligent Systems, 2(3), 161–173. https://doi.org/10.47852/BONVIEWJDSIS32021386

Vaibhav Suman. (2023). HF dataset.

Vigneshwer Dhinakaran. (2018, November 13). Correlation matrix for EDA.

Ydata. (2023, June 9). Imbalance Dataset.
 

