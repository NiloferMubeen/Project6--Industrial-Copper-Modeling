# Industrial Copper Modeling
Industrial Copper modeling is a project that involves building predictive models for both regression (predicting Selling Price) and classification (Status Prediction- Won or Lost), 
as well as creating an interactive web application using Streamlit for model deployment.

# Tools Required
1. Python
2. Pandas
3. Numpy
4. Scikit Learn
5. Seaborn
6. Matplotlib
7. Streamlit

# Approach
### 1. Data Preprocessing
   The dataset contains missing values that need to be addressed. These null values are handled through mode imputation. The categorical columns are Encoded using the
   Label encoder for efficien usage in model building. The outliers are identified using the boxplots and are handled with IQR method. Most of the columns are right skewed and are handled using
   Box Cox Transforms. These encoded values are then inverse transformed for user-friendly application in Streamlit
### 2. EDA:
   The skeness and outliers in the data are visualized using the libraries like Matplotlib and Seaborn 
### 3. Model Building:
## Classification :

Success and Failure Prediction: In this classification, the 'status' variable is used, which defines Transaction or item status as 'Won' for Success and 'Lost' for  Failure. 
Data points with status values other than 'Won' and 'Lost' are excluded from our dataset.

Imbalance in Target Variable: Imbalance within the 'status' variable is handled using  SMOTE (oversampling method).

Model Selection: The dataset is divided into training and testing subsets, for our classification and different algorithms are applied to assess the performance. After evaluation, the Random Forest Classifier model was chosen for its ability to efficiently distinguish between the positive and negative classes with an accuracy of 96%.

## Regression:

Price Prediction: The primary objective in this regression task is to predict the selling price which is a continous variable.

Model Selection: The dataset is divided into training and testing subsets, for our regression Task and different algorithms are applied to assess the performance. The Random Forest Regression model explains or predicts 95% of the relationship between the dependent and independent variables. Hence I chose the Ransdom Forest Regression Model

