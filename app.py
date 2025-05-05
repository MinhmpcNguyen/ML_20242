import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("/Users/Yuki/ML_20242/best_cnn_model_50epoch.h5")


class_names = ["Cat", "Dog"]  # 0 = Cat, 1 = Dog


def predict(input_image):
    # Resize về đúng kích thước đầu vào của model
    image = input_image.resize((200, 200))

    # Chuyển ảnh thành numpy array, normalize và reshape
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 200, 200, 3)  # Batch size = 1

    # Dự đoán
    prediction = model.predict(image_array).flatten()[0]  # Xác suất duy nhất
    label = class_names[int(prediction > 0.5)]

    return f"{label} ({prediction:.2f} confidence)"


# Tạo Gradio UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Cat/Dog Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Cat vs Dog Classifier",
    description="Upload an image of a cat or dog. Model will predict the class.",
)

if __name__ == "__main__":
    demo.launch()
