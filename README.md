# Từ thư mục ML_20242/
docker build -t catdog-app .

docker run -p 7860:7860 catdog-app

## If you want to use VGG-19 Model to predict an Image
- Step 1: Download the model in this [Link](https://drive.google.com/drive/folders/1cRM4CP9K2FG31VSo-J68QcQi-XWpZwEA?usp=drive_link)
- Step 2: Do the same like the others
