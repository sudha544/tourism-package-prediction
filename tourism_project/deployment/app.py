import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the tourism package prediction model
model_path = hf_hub_download(repo_id="sudha1726/tourism_package_model", filename="tourism_package_bestmodel.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package based on their profile and preferences.
Please enter the customer details below to get a prediction.
""")

# User input fields
type_of_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Website", "Partner", "Others"])
gender = st.selectbox("Gender", ["Male", "Female"])
city_tier = st.selectbox("City Tier", ["1", "2", "3"])
occupation = st.selectbox("Occupation", ["Salaried", "Business", "Retired", "Student", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "Others"])
product_pitched = st.selectbox("Product Pitched", ["Standard", "Gold", "Platinum", "Silver"])
own_car = st.radio("Own Car", ["Yes", "No"])
passport = st.radio("Passport", ["Yes", "No"])

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=80, value=30)
monthly_income = st.number_input("Monthly Income (in ₹)", min_value=1000, max_value=1000000, value=10000)
number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=0, max_value=20, value=5)
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0, max_value=10, value=0)
preferred_property_star = st.number_input("Preferred Property Star", min_value=1.0, max_value=5.0, value=5.0)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
number_of_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=4)

# Encoding mappings
type_of_contact_map = {"Self Enquiry": 0, "Company Website": 1, "Partner": 2, "Others": 3}
gender_map = {"Male": 0, "Female": 1}
city_tier_map = {"1": 0, "2": 1, "3": 2}
occupation_map = {"Salaried": 0, "Business": 1, "Retired": 2, "Student": 3, "Other": 4}
marital_status_map = {"Single": 0, "Married": 1}
designation_map = {"Manager": 0, "Executive": 1, "Senior Manager": 2, "Others": 3}
product_pitched_map = {"Standard": 0, "Gold": 1, "Platinum": 2, "Silver": 3}
own_car_map = {"Yes": 1, "No": 0}
passport_map = {"Yes": 1, "No": 0}

# Assemble input data into DataFrame
input_data = pd.DataFrame([{
    'TypeofContact': type_of_contact_map[type_of_contact],
    'Gender': gender_map[gender],
    'CityTier': city_tier_map[city_tier],
    'Occupation': occupation_map[occupation],
    'MaritalStatus': marital_status_map[marital_status],
    'Designation': designation_map[designation],
    'ProductPitched': product_pitched_map[product_pitched],
    'OwnCar': own_car_map[own_car],
    'Passport': passport_map[passport],
    'Age': age,
    'MonthlyIncome': monthly_income,
    'NumberOfTrips': number_of_trips,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfFollowups': number_of_followups
}])

# Prediction
if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase" if prediction == 1 else "Not Likely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

    st.write("---")
st.subheader("Upload Deployment to Hugging Face Space")

if st.button("Upload Deployment Files"):
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        st.error("HF_TOKEN is not set. Cannot upload files.")
    else:
        login(token=HF_TOKEN)
        api = HfApi()
        repo_id = "sudha1726/tourism-package-prediction"  # your Space repo
        folder_path = "tourism_project/deployment"

        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="space",
            path_in_repo=""
        )
        st.success("Deployment files uploaded successfully!")
