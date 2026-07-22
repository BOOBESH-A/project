# Instagram Filter Recommendation System

A Machine Learning-based Desktop Application built with Python, OpenCV, Scikit-learn, and CustomTkinter.

## Objective
Analyze an uploaded image using computer vision and ML techniques to recommend the most suitable Instagram-style filters, providing previews and the ability to save the filtered output.

## Features
- Extract image features (brightness, contrast, saturation, hue, mood)
- Apply a weighted scoring model (ML-based rule scoring + KNN similarity)
- Rank filters by confidence score
- Render previews and allow save/compare actions

## Installation
Install the dependencies using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage
Run the application using:
```bash
python filter_app.py
```
