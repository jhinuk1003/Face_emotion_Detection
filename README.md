# Face_emotion_Detection
````markdown
# 🧠 Real-Time Emotion Detection App 😄😡😢😲

A real-time emotion detection web application using **Streamlit**, **OpenCV**, and a **deep learning model** trained to classify human emotions from facial expressions.

---

## 🚀 Features

- 🎥 Live webcam feed detection  
- 🧠 Real-time emotion classification (Happy, Sad, Angry, etc.)  
- 📦 Built with Keras, TensorFlow, and OpenCV  
- 🌐 Streamlit-based user interface  
- 🎯 Lightweight and beginner-friendly  

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- OpenCV  
- Keras / TensorFlow  
- Haarcascade Classifier

---

## 📸 How It Works

1. Captures live video using your webcam  
2. Detects faces using Haarcascade  
3. Passes face ROI to a CNN model  
4. Predicts and displays emotion on screen

---

## ▶️ Run Locally

```bash
git clone https://github.com/your-username/real-time-emotion-detection.git
cd real-time-emotion-detection
pip install -r requirements.txt
streamlit run streamlit_emotion_detection.py
````

📝 **Note:** Make sure you have a webcam connected and the model file `model.keras` & `haarcascade_frontalface_default.xml` in the same directory.

---

## 📂 File Structure

```
├── streamlit_emotion_detection.py
├── model.keras
├── haarcascade_frontalface_default.xml
├── requirements.txt
```

---

## 🎉 Screenshots


---

## 🤝 Contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📃 License

This project is open-source and free to use under the [MIT License](LICENSE).

---

## ❤️ Made with love & code by Jhinuk Roy

```

