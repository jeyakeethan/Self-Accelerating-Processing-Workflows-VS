import pickle
import time
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

loaded_model = pickle.load(open("dumped_svc_model", 'rb'))

def predictTime(args):
        return loaded_model.predict([args])
