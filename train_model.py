import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    data = pd.read_csv("insurance.csv")

    encoder = LabelEncoder()

    data["sex"] = encoder.fit_transform(data.sex)
    data["region"] = encoder.fit_transform(data.region)
    data["smoker"] = encoder.fit_transform(data.smoker)

    X = data.drop(columns="charges", axis=1)
    Y = data["charges"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=101
    )

    model = RandomForestRegressor()
    model.fit(X_train, Y_train)

    testing_data_prediction = model.predict(X_test)
    score = metrics.r2_score(Y_test, testing_data_prediction)
    print("R2 score:", score)

    filename = "medical_insurance_cost_predictor.sav"
    pickle.dump(model, open(filename, "wb"))
    print("Saved model to:", filename)

    input_data = (19, 0, 27.9, 0, 1, 3)
    input_data_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_array)
    print("Predicted Medical Insurance Cost:", prediction)


if __name__ == "__main__":
    main()
