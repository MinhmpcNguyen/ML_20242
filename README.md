

# Cat-Dog Classification Demo

This project demonstrates multiple machine learning models applied to the Cat-Dog classification problem.

## How to Run the Demo

From the root folder `ML_20242/`, run the following commands:

```bash
docker compose -f compose.yaml build
docker compose -f compose.yaml up
```

These commands will build and start the Docker environment required for the demonstration.

**Note:**  
Due to the size of the ANN and VGG models, they will be downloaded from Hugging Face during the setup process. Additionally, some dependencies are large and may take time to install. Please expect the setup process to take approximately **7–10 minutes** depending on your system and internet connection.

## Included Models

The demo includes the following models:

- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest
- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- VGG
- Inception V3

Each model is integrated to classify images of cats and dogs with varying levels of complexity and performance.

## Dependencies

All dependencies are managed and installed within the Docker container. No additional manual setup is required outside of Docker.


