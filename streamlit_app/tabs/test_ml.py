import pandas as pd
import joblib



MODEL_PATH = "../../models/prot_ml_1.joblib"

## Get User input for parameters

#Input   : Dataframe to test
#Returns : Dataframe with prediction column, accuracy score
def predict(df):
    #Open the model file
    model = joblib.load(MODEL_PATH)
    #make the prediction
    data = df
    if(['classification'] in df):
        data = df.drop('classification', axis=1)
    y_pred = model.predict(data)

    #insert predictions in provided array
    df['predicted_labels'] = y_pred

    return df
    