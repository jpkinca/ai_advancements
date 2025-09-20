# Earnings Call Analysis using NLP Vectorization
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import re
import warnings
warnings.filterwarnings('ignore')

# NLP and ML libraries
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from textblob import TextBlob

# Vector search
import faiss

class EarningsCallAnalyzer:
    def __init__(self):
        # Load pre-trained models
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.financial_model = None  # Will try to load FinBERT if available
        self.tfidf = TfidfVectorizer(max_features=500, stop_words='english')
        self.topic_model = LatentDirichletAllocation(n_components=10, random_state=42)
        
        # Load spaCy for NER and text processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Storage
        self.earnings_data = {}
        self.embeddings = {}
        self.sentiment_scores = {}
        self.topics = {}
        
    def load_financial_model(self):
        """Try to load FinBERT for financial text analysis"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from transformers import pipeline
            
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.financial_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
            print("FinBERT loaded successfully")
        except:
            print("FinBERT not available. Using general sentiment analysis.")
    
    def simulate_earnings_transcript(self, symbol, quarter, year):
        """
        Simulate earnings call transcript data
        In production, you'd integrate with services like:
        - Alpha Vantage (earnings call transcripts)
        - Financial data APIs
        - Web scraping from investor relations pages
        """
        # Simulate different types of earnings call content
        sample_transcripts = {
            'positive': [
                f"We are pleased to report strong quarterly results for {symbol}. Revenue grew 15% year-over-year, exceeding our guidance. Our margin expansion initiatives are delivering results, with gross margins improving 200 basis points. We see continued strength in our core markets and remain optimistic about future growth prospects. Customer demand remains robust across all segments.",
                f"This quarter represents a milestone for {symbol}. We achieved record revenue of $2.1 billion, driven by exceptional performance in our cloud division. Operating margins expanded significantly due to operational efficiency improvements. We are raising our full-year guidance and expect continued momentum in the coming quarters. Our strategic investments in R&D are paying off with innovative product launches."
            ],
            'negative': [
                f"While {symbol} faced some headwinds this quarter, we remain focused on our long-term strategy. Revenue declined 5% due to challenging market conditions and supply chain disruptions. We are implementing cost reduction measures and expect to see improvement in the second half of the year. Despite near-term challenges, our fundamental business remains strong.",
                f"This quarter was below our expectations for {symbol}. We experienced margin pressure due to increased competition and rising input costs. Customer acquisition slowed in key markets, and we are reassessing our pricing strategy. We are taking decisive action to address these challenges and position the company for future growth."
            ],
            'neutral': [
                f"{symbol} delivered results in line with expectations this quarter. Revenue grew 3% year-over-year, consistent with industry trends. We continue to invest in strategic initiatives while maintaining disciplined cost management. Market conditions remain mixed, but we are confident in our competitive position and long-term outlook.",
                f"Our Q{quarter} results for {symbol} reflect a stable operating environment. Revenue was flat compared to last year, with strength in some segments offset by weakness in others. We are focused on execution of our strategic plan and expect gradual improvement over time. Investment in digital transformation continues to be a priority."
            ]
        }
        
        # Simulate Q&A section
        qa_sections = [
            "Analyst: Can you provide more color on the margin trends? Management: We expect margins to stabilize as we lap the headwinds from last year.",
            "Analyst: What are you seeing in terms of customer demand? Management: Demand remains healthy but we are seeing some lengthening in sales cycles.",
            "Analyst: How should we think about capital allocation going forward? Management: We remain committed to returning cash to shareholders while investing in growth."
        ]
        
        # Randomly select transcript type based on recent stock performance
        try:
            stock = yf.Ticker(symbol)
            recent_data = stock.history(period='3mo')
            performance = (recent_data['Close'][-1] - recent_data['Close'][0]) / recent_data['Close'][0]
            
            if performance > 0.1:
                transcript_type = 'positive'
            elif performance < -0.1:
                transcript_type = 'negative'
            else:
                transcript_type = 'neutral'
        except:
            transcript_type = 'neutral'
        
        # Combine transcript sections
        main_transcript = np.random.choice(sample_transcripts[transcript_type])
        qa_section = " ".join(np.random.choice(qa_sections, 2))
        
        full_transcript = f"Prepared Remarks: {main_transcript} Q&A Section: {qa_section}"
        
        return {
            'transcript': full_transcript,
            'date': datetime(year, (quarter-1)*3 + 1, 1),
            'symbol': symbol,
            'quarter': quarter,
            'year': year
        }
    
    def extract_financial_entities(self, text):
        """Extract financial entities and metrics from text"""
        entities = {
            'revenue_mentions': [],
            'margin_mentions': [],
            'guidance_mentions': [],
            'risk_factors': [],
            'growth_drivers': [],
            'numbers': []
        }
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract monetary values and percentages
            for ent in doc.ents:
                if ent.label_ in ['MONEY', 'PERCENT', 'QUANTITY']:
                    entities['numbers'].append(ent.text)
            
            # Pattern matching for key financial terms
            revenue_patterns = r'\b(?:revenue|sales|top.?line)\b.*?(?:\$[\d,.]+ (?:billion|million)|[\d.]+%)'
            margin_patterns = r'\b(?:margin|profitability|gross|operating)\b.*?(?:\$[\d,.]+ (?:billion|million)|[\d.]+%)'
            guidance_patterns = r'\b(?:guidance|outlook|forecast|expect|project)\b.*?(?:\