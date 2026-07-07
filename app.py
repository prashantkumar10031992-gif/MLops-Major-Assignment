from flask import Flask, render_template, request
import joblib
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = joblib.load("model/savedmodel.pth")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    image = Image.open(file).convert("L")
    image = image.resize((64, 64))

    image = np.array(image).flatten() / 255.0
    image = image.reshape(1, -1)

    prediction = model.predict(image)

    return f"<h2>Predicted Class: {prediction[0]}</h2><br><a href='/'>Go Back</a>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)