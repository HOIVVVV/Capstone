import logging
from typing import Counter
from flask import Flask, request, jsonify, render_template, send_file
import openai
import os, io, librosa, torch, base64, librosa.display, zipfile
import matplotlib.pyplot as plt
from base64 import b64encode
import numpy as np
import seaborn as sns
import pandas as pd

# Use a non-GUI backend for Matplotlib
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# Path to the folder containing audio files
FOLDER_PATH = 'static/data/audiofiles'  # The folder containing your audio files

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model once when the app starts
#def load_model(model_path):
 #   return model

#MODEL_PATH = 'models/class_shipsEar.pth'
#model = load_model(MODEL_PATH)


if __name__ == '__main__':
    app.run(debug=True)
