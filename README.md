# Titanic_survival_app
Titanic ML Project - Summary Notes
 1. Objective:- Predict Titanic survival using ML models, trained in AWS SageMaker and used in Streamlit.
 2. Environment Setup:- Trained the model in SageMaker Studio.- Used pandas, scikit-learn, and XGBoost for data prep and modeling.
 3. Data Preprocessing:- Handled missing values: Age (median), Fare (median), Embarked (mode).- Dropped non-useful columns: Cabin, Ticket, Name.- One-hot encoded Embarked.- Converted Sex to numeric: male=0, female=1.- Added FamilySize = SibSp + Parch + 1.
 4. Models Trained:- XGBoost
 5. Model Saving & Deployment:- Saved trained model as 'titanic_model.pkl' using joblib.- Downloaded it from SageMaker for local use.
 6. Streamlit App:- Built app with user input (Pclass, Age, Sex, Fare, etc.)
- App loads model and predicts survival instantly.- Ran locally using: streamlit run app.py
7. Key Questions Answered:- What is .pkl? A saved model file (Python pickle format).- Why train on SageMaker? For big data, GPU, scalable ML pipelines.- Why not run Streamlit in SageMaker? It's a web server, which SageMaker blocks for security.- Best practice: Train in SageMaker, deploy as endpoint, connect from Streamlit.
8. Best Practice (Production):- Deploy model in SageMaker Endpoint.- Call endpoint via boto3 from any frontend (Streamlit, React, etc.)
 9. Final Output:- Functional ML-powered web app for Titanic survival prediction.- Built end-to-end workflow from training to deployment
