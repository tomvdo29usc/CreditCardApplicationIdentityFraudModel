<!-- ABOUT THE PROJECT -->
## About The Project

Identity fraud is the act of misrepresenting who they are when they apply for a product or service, causing businesses close to $56 billions in 2021. This project will tackle identity fraud in credit card application using intensive data analytics and advanced machine learning method to save companies from financial losses.

**Tools**: Python (NumPy, Pandas, Scikit-learn, PySpark), MS Excel, Spark

**Skills**: Exploratory Data Analysis, Applied Statistics, Data Visualization, Machine Learning, Feature Engineering, Fraud Analytics

# 1. Data Exploratory Analysis

The data is about **application for credit cards and cell phones with personal identifying information**. There are **10 fields and a million records** in the data. The data covers applications submitted in **the year of 2017.**

## 1.1 Summary Table

(1) Numerical Fields
![image](https://github.com/user-attachments/assets/44e1fa28-a499-499d-ba5d-d7121b39186c)

(2) Categorical Fields
![image](https://github.com/user-attachments/assets/a3e8e2b1-eee6-462c-a1fb-4983c18056de)

> [!NOTE]
> There is no missing value

## 1.2 Data Quality

### 1.2.1 Field “SSN
![image](https://github.com/user-attachments/assets/287113a7-796a-45a9-8a08-61c9e3a810e2)

> [!NOTE]
> 1. Pad any SSN with less than 9 digit with leading 0s.
> 2. Replace frivolous SSN value “999999999” with a record number string to create a unique SSN to avoid false alarm risk

### 1.2.2 Field “firstname” and “lastname”
![image](https://github.com/user-attachments/assets/60892452-703c-456b-837a-bbf6d0e6308b)
![image](https://github.com/user-attachments/assets/6f61caba-9c6a-495e-b01f-e8793d8938d8)

> [!NOTE]
> There is no data quality issue with the field First Name and Last Name

### 1.2.3 Field “address”
![image](https://github.com/user-attachments/assets/3c32113e-c86f-47d8-bbee-c18b71cd3b41)

> [!NOTE]
> The address value “123 MAIN ST” appears to be the placeholder or frivolous value, accounting for 0.1079% of the data but appearing 1,012.37% more than the second most popular address in the data. We don't want this value to be a false alarm to the model. Therefore, replace them with a distinct record number to create a unique address for an application.

### 1.2.4 Field “zip5”
![image](https://github.com/user-attachments/assets/c0ff1caa-11c0-42b7-81fd-5d6fa7182b98)

> [!NOTE]
> Pad any Zip5 with less than 5 digit with leading 0s.

### 1.2.5 Field “dob”
![image](https://github.com/user-attachments/assets/8d72ef1e-0743-4f2d-83dc-a91b58f109dc)

> [!NOTE]
> The date of birth value '19070626' appears to be the placeholder or frivolous value, accounting for 12.6568% of the data and appearing 2,526.98% more than the second most popular date of birth in the data. We don't want this value to be a false alarm to the model. Therefore, assign them a distinct value for each of those applications.

### 1.2.6 Field “homephone”
![image](https://github.com/user-attachments/assets/bc848a3b-d3ab-44ba-a4f1-bc8d7a901211)

> [!NOTE]
> 1. Pad any homephone with less than 10 digit with leading 0s.
> 2. Replace frivolous homephone value “9999999999” with a record number string to create a unique homephone to avoid false positive risk

## 1.3 More Data Visualizations
![image](https://github.com/user-attachments/assets/d0ad6d94-6000-48a9-8bd1-fe05d9a46a83)
![image](https://github.com/user-attachments/assets/64117837-7b0d-4946-b4c0-2e27d789839e)
![image](https://github.com/user-attachments/assets/19c408f8-6359-4160-a4d6-c4bad025e1b1)

> [!NOTE]
> - The normalized daily good applications percentage seems to be more stably constant than the normalized daily bad applications percentage. 
> - Fraud rate is higher later of the week (Wed-Sun) than the early of the week (Mon-Wed)
> - It’s interesting that many applicants has the age above 110. In fact, the number of applicants within the age group is more than double compared to other age groups. However, the % of fraud applications are quite evenly stable, indicating that the data is not biased fraud towards old applicants even intuitively applicants of that age is quite rare. 

# 2. Features Engineering

Signal of Identity Fraud:

1. **Fraudster originating many applications from the same place**: Unusual number of applications with the same address and/or phone number
2. **A compromised identity, perhaps available for purchase**: A particular SSN, Name_DOB combination being used with many different addresses, phones combinations

Fraud Modes:

1. An individual fraudster has gotten a list of identity information (stolen, bought on internet…) and is going through this list applying for many products with many identities. He uses the victims’ core identity information (SSN,Name,DOB) and his own contact information (Address, Phone number).
    - How many applications have we seen in the recent past that have that same address or phone number?
2. A victim’s identity was compromised in a data breach and his core identity information (SSN,Name,DOB) is being used by many fraudsters.
    - How many applications have we seen in the recent past that have that SSN or Name_DOB or both?

Approaches:
![image](https://github.com/user-attachments/assets/7ec17f63-b9af-4408-857d-e85d65305b7a)

> [!NOTE]
> In total, generated more than 4,000 features tailored to identity theft dynamics, equipping model with targeted insights to effectively flag fraudulent applications.

# 3. Features Selection

Previously generated 4,000+ variables posed high dimensionality and potential multi-collinearity challenges. Our goal was removing weak predictors to minimize dimensionality while retaining predictive power. Here is the process of features selection:
![image](https://github.com/user-attachments/assets/dab5d190-45bb-457d-af44-ca6f0c027805)

> [!NOTE]
> We measured the model performance using this process: The model scores the fraud scores for all application. We select 3% of applications with the highest score as fraud and measure the accuracy of fraud application captured within that 3% pool. 


Below is the best performance by number of features:
![image](https://github.com/user-attachments/assets/aec474ac-2682-439c-92d1-8a0bed666b3e)

Below is the the final optimal selected set of features:
![image](https://github.com/user-attachments/assets/4935ce51-d2a7-490d-a4b5-b29d6da18c8d)

# 4. Model Development

## 4.1 Preliminary Model Exploration

We developed and evaluated 7 Machine Learning Classification models, starting with simple and highly interpretable model like Logistic Regression. Then we experimented with non-linear models and adjusted its hype-parameters as well as number of variables to explore bias and variance behaviors. To develop the model, we split the data into train/test set which is the first 10 months and Out-of-time set which is the last 2 months. Each model was trained and evaluated on randomly split train/test set, in the meantime, recording the performance on Out-of-time, for 10 times. Eventually, models were fine-tuned to achieve the highest performance on the Out-of-time set. 

The baseline model was Logistic Regression which achieved the average Fraud Detection Rate (FDR) of approximately **51.53%** at 3% cutoff rate on Out-of-time (OOT) set. One of the best models can achieve slightly above **55%**. Below is the performance result of 7 models:
![image](https://github.com/user-attachments/assets/07087ae3-914c-4221-8d46-c1f200072fb3)
![image](https://github.com/user-attachments/assets/d9f9f74a-106d-4b27-af35-b20c38b938bd)

## 4.2 Final Model Performance

Neural Network and Random Forest were amongst the model with high performance on Out-of-time. Neural Network performed a slightly better than the other; however, due to training a Neural Network model was more time-consuming, we didn’t select this model. Therefore, we selected Random Forest as the final model:

<img width="461" alt="image" src="https://github.com/user-attachments/assets/9fb2c90e-c279-4730-a946-4e9ebb9cc9b9">

Below is the performance results of the best model:
![image](https://github.com/user-attachments/assets/1f6ab29f-aa76-4191-969d-4accad424de2)
![image](https://github.com/user-attachments/assets/fc4a058f-c08e-4d33-aa47-f02d3974df6e)
![image](https://github.com/user-attachments/assets/49e39783-58f3-423d-9b0c-ea3b184c0570)

> [!NOTE]
> The best model was able to achieve 55.07% of Fraud Detection Rate while only rejecting the top 3% of all applications.

# 5. Estimate Impacts

Experian data from Q3 2022 shows American consumers had an average total credit limit of **$28,930** across all revolving credit accounts. As of January 2024, the average American spends more than **$1,500 per month** on their credit card. Assume these numbers are true, if a fraudster successfully opens a credit card account, the company will lose **$28,930** from the fraudster’s illegal spendings. However, if the legit customers wanted to open a credit line but was rejected due to False Positive error of the Fraud model, the company would lose the potential income from the customer’s credit card usage. We assume an average customer would spend **$1,500** per month while the company makes **2.5%** from the spending fee. J.D. Power found that 51% of Americans can't pay off their entire balance each month and instead let it revolve to the next month. Also, let assume the credit card APR is **22%** (or **1.83%** monthly). We estimate that the credit card company would lose **$563.66/year** per false positive error. Also, we assume most customers would use the card for **2 years** before they want to cancel. Below is our estimate savings using the fraud model:
![image](https://github.com/user-attachments/assets/842de46b-cad2-42c8-8235-6fbf09ea237e)

> [!NOTE]
> We suggest using the model at the Fraud Cutoff Rate at 2% to achieve the optimal annual savings of $34,919,012.56 per 1 million applications with around 1.44% of fraud (from this dataset) according to above assumptions. 








