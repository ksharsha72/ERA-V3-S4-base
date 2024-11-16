from flask import Flask, render_template, jsonify
import json
from pathlib import Path

app = Flask(__name__)

# Global variables to store metrics
metrics = {
    "train_loss": [],
    "train_acc": [],
    "current_epoch": 0,
    "current_loss": 0,
    "current_acc": 0,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_metrics")
def get_metrics():
    return jsonify(metrics)


@app.route("/clear_metrics")
def clear_metrics():
    metrics["train_loss"] = []
    metrics["train_acc"] = []
    metrics["current_epoch"] = 0
    metrics["current_loss"] = 0
    metrics["current_acc"] = 0
    return jsonify({"status": "success"})


@app.route("/update_metrics/<epoch>/<loss>/<acc>")
def update_metrics(epoch, loss, acc):
    # If it's the first update of epoch 0, clear previous metrics
    if int(epoch) == 0 and len(metrics["train_loss"]) > 0:
        metrics["train_loss"] = []
        metrics["train_acc"] = []

    metrics["current_epoch"] = int(epoch)
    metrics["current_loss"] = float(loss)
    metrics["current_acc"] = float(acc)
    metrics["train_loss"].append(float(loss))
    metrics["train_acc"].append(float(acc))
    return jsonify({"status": "success"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
