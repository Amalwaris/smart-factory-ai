import joblib

def load_models():
    clf = joblib.load("app/models/classifier.pkl")
    anomaly_model = joblib.load("app/models/anomaly_detector.pkl")
    label_encoder = joblib.load("app/models/encoder.pkl")
    return clf, anomaly_model, label_encoder
