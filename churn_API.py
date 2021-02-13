import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np

#The function will create the derived variable needed for model score..
def create_cpm(df):
    charge_vars = [x for x in df.columns if 'charge' in x]
    minutes_vars = [x for x in df.columns if 'minutes' in x]
    df['total_charges'] = 0
    df['total_minutes'] = 0
    for indexer in range(0, len(charge_vars)):
        df['total_charges'] +=  df[charge_vars[indexer]]
        df['total_minutes'] +=  df[minutes_vars[indexer]]
    df['charge_per_minute'] = np.where(df['total_minutes'] >0, df['total_charges']/df['total_minutes'], 0)
    df.drop(['total_minutes', 'total_charges' ], axis = 1, inplace = True)
    return df

# load model
model = pickle.load(open('model.pkl','rb'))

#load column order..
model_columns = pickle.load(open('model_columns.pkl','rb'))


# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    #adding the required derived and modified columns
    data_df = create_cpm(data_df)
    data_df['international_plan_num'] = data_df['international_plan'].apply(lambda x : 1 if x == 'yes' else 0)
    data_df = data_df.reindex(columns=model_columns, fill_value=0)#added by SS

    # predictions
    result = model.predict_proba(data_df)[:, 1]
    result2 = 0
    if result[0] >=0.5:
        result2 = 'yes'
    else:
        result2 = 'no'

    # send back to browser
    output = {'Model_score': result[0],  'Churn' : result2, 'cpm' : data_df['charge_per_minute'][0]  }

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)