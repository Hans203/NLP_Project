# Aspect-Based Sentiment Analysis for Car Reviews

This project is a **Streamlit web application** that performs **Aspect-Based Sentiment Analysis (ABSA)** on car reviews. 
It helps companies understand customer feedback by breaking down reviews into aspects (e.g., Engine, Mileage, Comfort, Price) 
and identifying whether the sentiment is **positive** or **negative**. Additionally, it uses **Google Gemini LLM** to 
summarize the most common **problems customers face**.

---

## 🚀 Features
- **LDA Based Aspects**: Engine/Performance, Mileage/Fuel, Comfort/Interior, Design/Style, Price/Value, Service/Maintenance.
- **Sentiment Analysis**: Uses **TextBlob** to classify sentences into positive or negative.
- **Aspect Detection**: Uses a pretrained **Word2Vec (CBOW) model** to map sentences to aspects.
- **Demo Mode**: Explore precomputed summaries for 5 brands.
- **Upload Mode**: Upload your own CSV file of car reviews for live analysis.
- **LLM Summarization**: Uses **Gemini (Google Generative AI)** to summarize negative feedback into actionable insights.

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Hans203/NLP_Project.git
cd NLP_Project
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK resources
```python
import nltk
nltk.download('punkt')
```

---

## 📂 Project Structure
```
├── app.py                     # Main Streamlit application
├── NLP_Project.ipynb          # Data Preprocessing, Visualisation and models used
├── car_reviews_cbow.model     # Pretrained Word2Vec model (CBOW)
├── summary.csv                # Precomputed sentiment summary (for demo mode)
├── brand_problems_summary.csv # Precomputed LLM problem summaries (for demo mode)
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory and add your **Google API Key** for Gemini:
```
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Running the App

Run the following command:
```bash
streamlit run app.py
```

The app will open in your browser at:  
👉 http://localhost:8501/

---

## 📊 Usage

### Demo Mode
- Shows **sentiment summary** and **problem summaries** for 5 preloaded car brands.

### Upload Mode
1. Upload a CSV file with a column `review` containing customer reviews.
2. Enter the brand name.
3. The app will:
   - Classify reviews into aspects & sentiments.
   - Extract negative reviews.
   - Generate summaries of main customer problems using Gemini.

---

## 📦 Example Input (CSV)
```csv
review
"The engine is powerful but the mileage is low."
"Seats are comfortable but the price is too high."
"Design looks modern and stylish."
"Service is very poor and costly."
```

---

## 📌 Requirements
- Python 3.8+
- Streamlit
- Pandas
- NLTK
- Gensim
- TextBlob
- python-dotenv
- google-generativeai

---

