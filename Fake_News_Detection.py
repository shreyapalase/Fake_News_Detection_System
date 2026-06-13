import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from IPython.display import display, HTML

# ===============================
# 1. BALANCED DATASET (IMPORTANT)
# ===============================
data = {
    "text": [
        # REAL
        "Government launches free healthcare scheme for citizens",
        "Scientists discover water on Mars using rover data",
        "Stock market rises due to strong economic growth",
        "Doctors confirm exercise improves mental health",
        "India launches new satellite successfully",
        "University develops low cost water purifier system",
        "Heavy rain alert issued by meteorological department",

        # FAKE
        "Aliens land secretly in India and meet government officials",
        "Drinking hot water cures all diseases instantly",
        "Crow becomes sparrow after eating magical fruit",
        "Moon is a hologram created by NASA",
        "Humans can fly after drinking special tea",
        "Time travel machine discovered in village laboratory",
        "Mobile phones can read human thoughts"
    ],
    "label": [
        "REAL","REAL","REAL","REAL","REAL","REAL","REAL",
        "FAKE","FAKE","FAKE","FAKE","FAKE","FAKE","FAKE"
    ]
}

df = pd.DataFrame(data)

# ===============================
# 2. TRAIN MODEL (IMPROVED)
# ===============================
X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

# ===============================
# 3. PREDICTION FUNCTION (WITH CONFIDENCE)
# ===============================
def predict_news(text):
    vec = vectorizer.transform([text])
    
    prediction = model.predict(vec)[0]
    prob = model.predict_proba(vec).max() * 100

    if prediction == "REAL":
        return f"<h3 style='color:green;'> REAL NEWS ({prob:.2f}% confidence)</h3>"
    else:
        return f"<h3 style='color:red;'> FAKE NEWS ({prob:.2f}% confidence)</h3>"

# ===============================
# 4. SIMPLE GUI WITH INSTRUCTIONS
# ===============================
html_code = """
<h2>Fake News Detection System</h2>

<p style="color:blue;">
👉 Enter a news sentence below (example formats given)
</p>

<input id="newsInput" style="width:500px;height:35px;" 
placeholder="Example: Scientists discover water on Mars">

<br><br>

<button onclick="predict()" style="padding:10px;background:green;color:white;">
Check News
</button>

<div id="output"></div>

<hr>

<h3> Try These (Click to auto-fill):</h3>

<ul>
<li onclick="setText(this)">Government launches free healthcare scheme</li>
<li onclick="setText(this)">Scientists discover water on Mars</li>
<li onclick="setText(this)">Aliens land secretly in India</li>
<li onclick="setText(this)">Moon is a hologram created by NASA</li>
</ul>

<script>
function predict(){
    var text = document.getElementById("newsInput").value;
    google.colab.kernel.invokeFunction('notebook.predict', [text], {});
}

function setText(el){
    document.getElementById("newsInput").value = el.innerText;
}
</script>
"""

display(HTML(html_code))

# ===============================
# 5. BACKEND CONNECTION
# ===============================
from google.colab import output

def backend(text):
    display(HTML(predict_news(text)))

output.register_callback('notebook.predict', backend)
