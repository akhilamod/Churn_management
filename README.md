# Churn_management

## Motivation
Telecom sector has become one of the most aggressive sectors in India, especially after the entry of Reliance Jio in 2016. Persence of multiple companies with competitive pricing has given several options to the consumers to choose from. Hence it becomes important for companies to monitor and predict attrition / churn rate in order to have a proactive and responsive plan of action to retain a significant market share. This is where churn prediction model fits in.

## Objective
- Create a Machine Learning model that predicts customer churn at a fictious Telecom Company
- Identify the key reasons behind a customer leaving the company
- Develop an incentive plan for enciting would-be churners to remain with the company

## Reasons behind attrition and Incentive plan

![image](https://user-images.githubusercontent.com/86396532/124375714-40451480-dcc1-11eb-8801-fd691dd4cf9c.png)


1. Tenure:
    - Inference - Majority of the people left during initial days of their tenure
    - Reason - First impression of service provided might not have been great
    - Sol - Give incentives (eg free internet/calls during night) to prevent people from leaving in the initial days
2. Contract:
    - Inference - Most people who left were on Month-to-Month contract
    - Reason - short-term contract allows more flexibility in switching telecom network
    - Sol - Make one/two year contracts more attractive (cheaper) than month-to-month contract (eg seen in Gym membership charges)
3. Payment Method:
    - Inference - Most people who left used electronic check as payment mode
    - Reason - Electronic check might not get credited instantly or acknowledged immediately by service provider
    - Sol - Offer cashbacks eg tie-up with digital payment companies like PayTm to provide vouchers / cashbacks on payment
4. Internet service:
    - Inference - Most people who left were using Fibre optic as Internet service
    - Reason - May be due to poor service delivery (slow internet/poor maintenance)
    - Sol - Improve the service quality in this area eg prompt greivance redressal

## Technical Components
- Logistic Regression has been used to create the churn model
- The model took 20 input parameters that comprised of tenure, monthly charges, different services that consumer availed, etc.
- Its performance was compared with other algorithms like Random Forest Classifier 
- Approach
  1. Data cleaning - removing NaNs, outliers
  2. Encoding - converting categorical variable into numerical values
  3. Checking Multicollinearity - to find relationship amongst independent variables 
  4. Checking correlation - to find level of impact an independent variable has over output
  5. Feature selection - which parameters determine the output
  6. Model Creation
  7. Performance 

### 1. Data cleaning
First dataset was checked for any null values and outliers. None of the input parameters showed any outliers

![image](https://user-images.githubusercontent.com/86396532/124375097-e727b180-dcbd-11eb-9f41-1676e49cfabb.png)

Statistical distribution of continuous variables was also checked for presence of any skewness

![image](https://user-images.githubusercontent.com/86396532/124375121-07f00700-dcbe-11eb-928f-d4996eabfd0b.png)

### 2. Encoding
Used One_hot_encoding for conversion of categorical variables into numeric values. For this used get_dummies option present in Pandas library

### 3. Multicollinearity
Multicollinearity in dataset means that the independent input variabels are having some relationship amongst themselves. Not removing it doesn't create any error in the model per se but it does increase redundancy. Meaning the info that is attained from a variable having high multicollinearity is already being provided by other variables, hence adding that variable into the model only reduces the speed without increasing accuracy. Multicollinearity was found using VIF (variance_inflation_factor) 

Result: Variables having VIF value > 10 were dropped 
  -  MonthlyCharges - VIF value: 210
  -  TotalCharges - VIF value: 17.7
  -  PhoneServie - VIF value: 9.3 (though <10 but low correlation with Churn as well ~ 0.012)

### 4. Correlation 
Checking correlation of input variables with output (Churn) is important as it helps remove those parameters that don't have any significant relation/impact over output. This reduces the burden on our model, makes it more efficient and accurate.
Result:
- Low correlation if abs(corr) < 0.09
    - gender_Male : -0.008612
    - PhoneService_Yes : 0.011942
    - MultipleLines_Yes : 0.040102
    - OnlineBackup_Yes : -0.082255
    - StreamingTV_Yes : 0.06
    - StreamingMovies_Yes : 0.06
- Key parameters that predict churn (High Correlation)
    - Tenure : -0.352229
    - InternetService_Fiber optic : 0.308020
    - Contract_Two year : -0.302253
    - PaymentMethod_Electronic check : 0.301919
- Though variables like MonthlyCharges, TotalCharges also have some correlation with Churn but they have high collinearity with other variables
- Their impact over Churn is captured by other independent variables hence are not included in key parameters to predict Churn

![image](https://user-images.githubusercontent.com/86396532/124375875-e2fd9300-dcc1-11eb-82a8-1cc3d1da883e.png)

### 5. Feature Selection
- Columns showing high multicollinearity (using VIF) and those showing low correlation with output (Churn) are dropped
- Dropping columns showing high multicollinearity : 
    - 'MonthlyCharges', 'TotalCharges'
- Dropping columns having low correlation with Churn : 
    - 'PhoneService_Yes', 'gender_Male', 'PhoneService_Yes', 
    - 'MultipleLines_Yes', 'OnlineBackup_Yes', 'StreamingTV_Yes', 'StreamingMovies_Yes'

### 6. Model Creation
Logistics regression was used for this classification problem as its much faster and has low computation cost. Hyperparameter tuning was done using RandomizedSearchCV to enhance performance of model

### 7. Performance 
- Since Telecom Company is more concerned about stopping people from leaving, hence its the False Negative (Type-II error) which is our biggest concern
- False negative means that the model declared those person as 'Not leaving' who were actually leaving the company
- The best parameter to measure False Negative (Type-II error) is Recall
- Hence the objective was to increase Recall value while keeping the precision value stable
- Over-sampling the train dataset helped increase Recall though at a slight cost of precision (f1_score remained stable)
- Final Result
    - Accuracy = 77.5%
    - Recall = 71%
    - Precission = 58%
    - F1_score = 64%

### Comparison with RandomForestClssifier algorithm
- The performance of logistic regression was compared with Random Forest Classifier using ROC curve 
- Result: Logistic regression outperformed Random Forest Classifier

![image](https://user-images.githubusercontent.com/86396532/125811903-55446f4d-d4c8-4361-be7f-4dad534fc631.png)


