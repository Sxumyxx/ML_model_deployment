from flask import Flask, render_template, request
import pickle

# Load tokenizer and model with correct paths
tokenizer = pickle.load(open(r"ML_model_deployment/models/cv.pkl", "rb"))
model = pickle.load(open(r"ML_model_deployment/models/clf.pkl", "rb"))

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get("content")

    if not email_text:
        return render_template("index.html", predictions="No input provided", email_text="")

    # Fix: Pass email_text inside a list to tokenizer.transform()
    tokenized_email = tokenizer.transform([email_text])

    # Make prediction
    predictions = model.predict(tokenized_email)[0]  # Extract first element

    # Convert prediction to 1 (spam) or -1 (ham)
    predictions = 1 if predictions == 1 else -1

    return render_template("index.html", predictions=predictions, email_text=email_text)

if __name__ == "__main__":
    app.run(debug=True)
