import requests

url = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint
																			

body = {
    'mean radius':17.99,
 'mean texture': 10.38,
 'mean perimeter': 122.8,
 'mean area' : 1001,
 'mean smoothness':0.1184,
 'mean compactness':0.2776,
 'mean concavity':0.3001,
 'mean concave points':0.1471,
 'mean symmetry': 0.2419,
 'mean fractal dimension': 0.07871,
 'radius error': 1.095,
 'texture error': 0.9053,
 'perimeter error': 8.589,
 'area error': 153.4,
 'smoothness error' : 0.006399,
 'compactness error': 0.04904,
 'concavity error': 0.05373,
 'concave points error': 0.01587,
 'symmetry error': 0.03003,
 'fractal dimension error': 0.006193,
 'worst radius' : 25.38,
 'worst texture': 17.33,
 'worst perimeter': 184.6,
 'worst area': 2019,
 'worst smoothness': 0.1622,
 'worst compactness': 0.6656,
 'worst concavity': 0.7119,
 'worst concave points': 0.2654,
 'worst symmetry':0.4601,
 'worst fractal dimension' : 0.1189
}
response = requests.post(url, data=body)
if respone==1:
    print("benign")
else:
    print("1")
    
print(response.json())
