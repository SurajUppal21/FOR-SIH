from flask import Flask, request, jsonify
from fastai.vision.all import *
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)

learner = load_learner(r'FMA-SIH\flood_classifier.pkl')
categories = ["Normal","Flood"]

def infer_image(image_data):
    label, _, probabilities = learner.predict(image_data)
    if label == '0':
        print(f"The area shown in the image is not flooded with probability {probabilities[0]*100:.2f}%.")
        return label,probabilities[0]*100
    elif label == '1':
        print(f"The area shown in the image is flooded with probability {probabilities[1]*100:.2f}%.")
        return label,probabilities[1]*100
    else:
        print("Unknown label assigned to image.")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        # Get the image path from the request
        image_data = request.files['image'].read()

        # Make a prediction
        prediction, probabilities = infer_image(image_data)

        probabilities = probabilities.tolist()

        return jsonify({"prediction": prediction, "probability":probabilities})

    except Exception as e:
        return jsonify({"error": str(e)})
    
pathlib.PosixPath = temp

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)