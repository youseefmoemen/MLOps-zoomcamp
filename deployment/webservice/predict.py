import pickle
from flask import Flask, request, jsonify
import mlflow



URI = 'sqlite:////home/youseef/mlflow.db'
MODEL_ID = "models:/taxi-best/19"


mlflow.set_tracking_uri(URI)
mlflow.set_registry_uri(URI)
model = mlflow.pyfunc.load_model(MODEL_ID)





def prepeare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


app = Flask('duration-prediction')



@app.route('/predict', methods=['POST'])
def predict():
    ride = request.get_json()
    features = prepeare_features(ride)
    pred = model.predict(features)
    return jsonify({'duration': float(pred[0])})



def predict_endpoint():
    ride = request.get_json()   
    features = prepeare_features(ride)
    pred = predict(features)

    result = {
        'duration' : pred
    }

    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
