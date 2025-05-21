import gradio as gr
import joblib
import numpy as np
from tensorflow.keras.models import load_model


def safe_load_model(path):
    try:
        return load_model(path)
    except:
        print(f"Cannot load model: {path}")
        return None


def safe_load_joblib(path):
    try:
        return joblib.load(path)
    except:
        print(f"Cannot load file: {path}")
        return None


cnn_model = safe_load_model("best_cnn_model_50epoch.h5")
ann_model = safe_load_model("best_ann_model.h5")
vgg_model = safe_load_model("vgg_model.h5")
inception_model = safe_load_model("inceptionv3.h5")

svm_result = safe_load_joblib("svm_model.pkl")
knn_result = safe_load_joblib("knn_model.pkl")
rf_model = safe_load_joblib("rfc_model.pkl")

if svm_result and isinstance(svm_result, tuple):
    svm_model, svm_pca, svm_scaler = svm_result
else:
    svm_model = svm_pca = svm_scaler = None

if knn_result and isinstance(knn_result, tuple):
    knn_model, knn_pca = knn_result
else:
    knn_model = knn_pca = None

class_names = ["Cat", "Dog"]


def predict_cnn(img):
    if not cnn_model:
        return "CNN model not available."
    image = img.resize((200, 200))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 200, 200, 3)
    pred = cnn_model.predict(image_array).flatten()[0]
    return "Dog" if pred > 0.5 else "Cat"


def predict_ann(img):
    if not ann_model:
        return "ANN model not available."
    image = img.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 224, 224, 3)
    pred = ann_model.predict(image_array).flatten()[0]
    return "Dog" if pred > 0.5 else "Cat"


def predict_vgg(img):
    if not vgg_model:
        return "VGG model not available."
    image = img.resize((224, 224))
    x = np.array(image) / 255.0
    x = x.reshape(1, 224, 224, 3)
    pred = vgg_model.predict(x).flatten()[0]
    return "Dog" if pred < 0.5 else "Cat"


def predict_inception(img):
    if not inception_model:
        return "InceptionV3 model not available."
    image = img.resize((224, 224))
    x = np.array(image) / 255.0
    x = x.reshape(1, 224, 224, 3)
    pred = inception_model.predict(x).flatten()[0]
    return "Dog" if pred > 0.5 else "Cat"


def predict_svm(img):
    if not all([svm_model, svm_pca, svm_scaler]):
        return "SVM model not available."
    image = img.resize((64, 64))
    x = np.array(image).flatten() / 255.0
    x = svm_scaler.transform([x])
    x = svm_pca.transform(x)
    pred = svm_model.predict(x)[0]
    return class_names[pred]


def predict_knn(img):
    if not all([knn_model, knn_pca]):
        return "KNN model not available."
    image = img.resize((64, 64))
    x = np.array(image).flatten() / 255.0
    x = knn_pca.transform([x])
    pred = knn_model.predict(x)[0]
    return class_names[pred]


def predict_rf(img):
    if not rf_model:
        return "Random Forest model not available."
    image = img.resize((64, 64))
    x = np.array(image).flatten() / 255.0
    pred = rf_model.predict([x])[0]
    return class_names[pred]


def predict_switch(img, selected_model):
    predictors = {
        "CNN": predict_cnn,
        "ANN": predict_ann,
        "VGG": predict_vgg,
        "InceptionV3": predict_inception,
        "SVM": predict_svm,
        "KNN": predict_knn,
        "Random Forest": predict_rf,
    }
    fn = predictors.get(selected_model)
    return fn(img) if fn else "Invalid model selection."


demo = gr.Interface(
    fn=predict_switch,
    inputs=[
        gr.Image(type="pil", label="Upload Cat/Dog Image"),
        gr.Radio(
            choices=[
                "CNN",
                "ANN",
                "VGG19",
                "InceptionV3",
                "SVM",
                "KNN",
                "Random Forest",
            ],
            label="Select Model",
        ),
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Cat vs Dog Classifier (Select Model)",
    description="Upload a cat/dog image and select which model to use for prediction.",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
