import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open("medical_insurance_cost_predictor.sav", "rb"))


# creating a function for Prediction
def medical_insurance_cost_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # reshape the array 
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]


def main():
    
    st.title("Medical Insurance Prediction Web App")

    # getting input from the user
    age = st.text_input("Age")
    sex = st.text_input("Sex: 0 -> Female, 1 -> Male")
    bmi = st.text_input("Body Mass Index")
    children = st.text_input("Number of Children")
    smoker = st.text_input("Smoker: 0 -> No, 1 -> Yes")
    region = st.text_input(
        "Region of Living: 0 -> NorthEast, 1-> NorthWest, 2-> SouthEast, 3-> SouthWest"
    )

    diagnosis = ""

    if st.button("Predicted Medical Insurance Cost:"):
        try:
            age_v = int(age)
            sex_v = int(sex)
            bmi_v = float(bmi)
            children_v = int(children)
            smoker_v = int(smoker)
            region_v = int(region)

            diagnosis = medical_insurance_cost_prediction(
                [age_v, sex_v, bmi_v, children_v, smoker_v, region_v]
            )
            st.success(diagnosis)
        except Exception:
            st.error("Please enter valid numeric values in all fields.")


if __name__ == "__main__":
    main()
