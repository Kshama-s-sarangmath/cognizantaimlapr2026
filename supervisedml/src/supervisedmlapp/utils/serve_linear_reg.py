#read linear regression pickle file
import numpy as np
import pickle
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'linear_regression_model.pkl')
   
#load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f'Model loaded from {model_path}')

#predict the price of a house with 3 bedrooms

area = 1500
predicted_price = model.predict(np.array([[area]]))
print(f'Predicted price for a house with {area} sqft: {predicted_price[0]}')