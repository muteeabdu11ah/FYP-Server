# Import flask and datetime module for showing date and time
from flask import Flask, request
from flask_cors import CORS
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import io

# # Get the directory path of the current script
# base_path = os.path.dirname(os.path.realpath(__file__))

# # Set the current working directory to the directory of the script
# os.chdir(base_path)


# Initializing flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

class Predictor:
    def __init__(self):
        try:
            self.model = load_model("lenet_fyp.h5")
        except Exception as e:
            print(e)

    def model_predict(self, img_path):
        test_image = image.load_img(img_path, target_size=(128, 128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        # test_image = test_image / 255.0
        result = self.model.predict(test_image)
        return result

pre = Predictor()

class_labels = ['Non Demented', 'Mild Demented', 'Moderate Demented', 'Very Mild Demented']

@app.route('/')
def home():
    return "This is the server for FYP Project, Send request on /data endpoint to get result"

# Route for seeing a data
@app.route('/data', methods=['POST', 'GET'])
def data():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        
        # Read the file
        img_bytes = file.read()
        # Use io.BytesIO to convert bytes to file-like object
        img = io.BytesIO(img_bytes)

        result = pre.model_predict(img)
        # Get the index of the predicted class
        predicted_class_index = np.argmax(result)
        # Get the predicted class name
        predicted_class = class_labels[predicted_class_index]

        accuracy = np.max(result)

        accuracy = round(float(accuracy * 100), 2)
        
        return {'predicted_class': predicted_class, 'accuracy': accuracy}
    
    return   "This is the server for FYP Project, Send request on /data endpoint to get result"

# Running app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
