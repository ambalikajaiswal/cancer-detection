from flask import Flask
from flask_restful import Api, Resource, reqparse
import joblib
import numpy as np

APP = Flask(__name__)
API = Api(APP)




class Predict(Resource):

    @staticmethod
    def post():
        BREAST_MODEL = joblib.load('model.pkl')
        parser = reqparse.RequestParser()
        cols_pars=joblib.load('model_columns.pkl')
        print(cols_pars)
        for i in cols_pars:
            parser.add_argument(i)
        args = parser.parse_args()  # creates dict

        X_new = np.fromiter(args.values(), dtype=float)  # convert input to array

        out = {'Prediction': BREAST_MODEL.predict([X_new])[0]}

        return out, 200


API.add_resource(Predict, '/predict')

if __name__ == '__main__':
    APP.run(debug=True)
