# Fake News Detection System (Machine Learning Project)

A simple Machine Learning-based web application that detects whether a given news statement is **REAL** or **FAKE** using Natural Language Processing (NLP) and Logistic Regression.

---

##  Project Overview

This project uses a small balanced dataset of real and fake news examples to train a text classification model. It converts text into numerical features using **TF-IDF Vectorization** and then applies **Logistic Regression** to classify the input.

The system also includes a simple interactive UI (Google Colab-based) where users can enter a news statement and get instant prediction with confidence score.

---

##  Features

-  Classifies news as REAL or FAKE  
-  Shows prediction confidence percentage  
-  Uses TF-IDF (N-gram based text features)  
-  Logistic Regression model  
-  Interactive UI using HTML + JavaScript (Colab compatible)  
-  Fast and lightweight ML pipeline  

---

## Tech Stack

- Python  
- Pandas  
- Scikit-learn  
- NLP (TF-IDF Vectorizer)  
- Logistic Regression  
- Google Colab (for UI integration)  
- HTML + JavaScript (basic frontend)

---

## Project Structure

```bash
Fake-News-Detection/
│
├── main.py (or notebook.ipynb)
├── README.md
└── requirements.txt (optional)
```


---

## ⚙️ How It Works

1. A dataset of labeled news (REAL / FAKE) is created
2. Text data is converted into numerical vectors using TF-IDF
3. Logistic Regression model is trained on the dataset
4. User input text is transformed using the same vectorizer
5. Model predicts whether news is REAL or FAKE
6. Confidence score is displayed

---

##  How to Run (Google Colab)

1. Open the notebook in Google Colab  
2. Run all cells sequentially  
3. Enter a news statement in the input box  
4. Click **"Check News"**  
5. View prediction result instantly  

---

## Example Inputs

Try these sample inputs:

- Government launches free healthcare scheme  
- Scientists discover water on Mars  
- Aliens land secretly in India  
- Moon is a hologram created by NASA  

---

##  Model Details

- Algorithm: Logistic Regression  
- Feature Extraction: TF-IDF (1-2 ngrams)  
- Train-Test Split: 70/30  
- Output: Binary classification (REAL / FAKE)

---

## Future Improvements

- Use larger real-world datasets  
- Implement Deep Learning models (LSTM / BERT)  
- Add web deployment (Flask / Django / Streamlit)  
- Improve UI design  
- Add multilingual support  

---

##  Disclaimer

This project is for **educational purposes only**.  
The dataset is very small and does not represent real-world performance.

---

## 👨‍💻 Author

**Name:** Shreya Sunil Palase 

**GitHub:** https://github.com/shreyapalase

---

If you like this project, don't forget to star the repository!
