# Trendyol Review Analyzer

Automated analysis tool for [Trendyol](https://www.trendyol.com/) product reviews that provides AI-powered summaries and sentiment insights through an interactive dashboard.


## Features

- Headless web scraping from Trendyol
- Filtering of non-product related reviews
- Sentiment analysis using BERT
- Review summarization using Gemini LLM


## Tech Stack

- **Scraping**: Selenium with Chrome
- **Sentiment Analysis**: [BERT model](https://huggingface.co/savasy/bert-base-turkish-sentiment-cased) trained for Turkish
- **Summarization**: Gemini - [Gemini API key](https://aistudio.google.com/app/apikey)
- **Frontend**: Streamlit
- **Visualizations**: Plotly


## Installation

1. Clone the repository
```bash
git clone https://github.com/enesmanan/review-summarizer.git
cd review-summarizer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

3. Run the application:
```bash
streamlit run app.py
```

-----


https://github.com/user-attachments/assets/e50d26c6-d432-4c9c-8e57-e0d6ec8b2e77


