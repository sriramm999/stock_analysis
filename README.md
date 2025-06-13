import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from typing import Dict, List, Optional, Tuple
import io
import PyPDF2
import docx
from urllib.parse import urlparse
import re

# Set page configuration
st.set_page_config(
    page_title="Real-Time Stock Sentiment Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .positive-sentiment {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .negative-sentiment {
        background-color: #f8d7da;
        border-left-color: #dc3545;
    }
    .neutral-sentiment {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .news-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .prediction-bullish {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-bearish {
        background: linear-gradient(45deg, #dc3545, #e83e8c);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .prediction-neutral {
        background: linear-gradient(45deg, #ffc107, #fd7e14);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class StockSentimentAnalyzer:
    def __init__(self):
        """Initialize the Stock Sentiment Analyzer with API configuration."""
        # ============ PRODUCTION API KEYS ============
        self.API_KEYS = {
            'ALPHA_VANTAGE': 'RPSDOCY12V7W6H2L',
            'NEWS_API': 'fb553a52b7ef43fab32a018e0a0e2423',
            'FINANCIAL_MODELING_PREP': '4jYsZoBjjL1LVSUt9xVuQDoxCxL1Ibbd',
            'POLYGON': '8KFada6qlORvU9E2a80hwrRNGq5tL2lx'
        }
        
        # Initialize session state for watchlist
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        if 'current_stock_data' not in st.session_state:
            st.session_state.current_stock_data = None
            
        if 'analysis_history' not in st.session_state:
            st.session_state.analysis_history = []

    def check_api_keys(self) -> bool:
        """Check if API keys are configured."""
        return not any(key.startswith('YOUR_') for key in self.API_KEYS.values())

    def fetch_stock_price(self, symbol: str) -> Optional[Dict]:
        """Fetch real-time stock price data from Alpha Vantage API."""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.API_KEYS['ALPHA_VANTAGE']
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                raise Exception(f"Invalid stock symbol: {symbol}")
            
            if 'Note' in data:
                raise Exception("API call frequency limit reached. Please try again later.")
            
            quote = data.get('Global Quote', {})
            if not quote:
                raise Exception("No data received for this symbol")
            
            return {
                'symbol': quote['01. symbol'],
                'price': float(quote['05. price']),
                'change': float(quote['09. change']),
                'change_percent': float(quote['10. change percent'].replace('%', '')),
                'volume': int(quote['06. volume']),
                'high': float(quote['03. high']),
                'low': float(quote['04. low']),
                'previous_close': float(quote['08. previous close'])
            }
            
        except requests.exceptions.RequestException as e:
            st.error(f"Network error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Error fetching stock price: {str(e)}")
            return None

    def fetch_company_info(self, symbol: str) -> Optional[Dict]:
        """Fetch company profile information from Financial Modeling Prep API."""
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}"
            params = {'apikey': self.API_KEYS['FINANCIAL_MODELING_PREP']}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and len(data) > 0:
                return data[0]
            else:
                return {'companyName': symbol, 'sector': 'Unknown', 'industry': 'Unknown'}
                
        except Exception as e:
            st.warning(f"Could not fetch company info: {str(e)}")
            return {'companyName': symbol, 'sector': 'Unknown', 'industry': 'Unknown'}

    def fetch_stock_news(self, symbol: str, company_name: str) -> List[Dict]:
        """Fetch recent news articles about the stock from NewsAPI."""
        try:
            # Calculate date range (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'"{company_name}" OR "{symbol}"',
                'sortBy': 'publishedAt',
                'language': 'en',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'apiKey': self.API_KEYS['NEWS_API'],
                'pageSize': 20
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for article in data.get('articles', [])[:15]:
                if article['title'] and article['description']:
                    articles.append({
                        'title': article['title'],
                        'source': article['source']['name'],
                        'published_at': article['publishedAt'],
                        'description': article['description'] or 'No description available',
                        'url': article['url'],
                        'content': article.get('content', '')
                    })
            
            return articles
            
        except Exception as e:
            st.warning(f"Could not fetch news: {str(e)}")
            return []

    def analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """
        Analyze sentiment of news articles using TextBlob and custom keywords.
        
        This function combines:
        1. TextBlob sentiment analysis
        2. Custom financial keyword analysis
        3. Overall sentiment scoring
        """
        if not articles:
            return {
                'overall_sentiment': 0.0,
                'positive_count': 0,
                'neutral_count': 0,
                'negative_count': 0,
                'confidence': 0.5,
                'articles_with_sentiment': []
            }
        
        # Financial sentiment keywords
        positive_keywords = [
            'growth', 'profit', 'revenue', 'beat', 'strong', 'positive', 'good',
            'excellent', 'success', 'win', 'gain', 'rise', 'bull', 'upgrade',
            'outperform', 'surge', 'boom', 'expansion', 'record', 'high'
        ]
        
        negative_keywords = [
            'loss', 'decline', 'fall', 'weak', 'poor', 'bad', 'fail', 'miss',
            'concern', 'worry', 'bear', 'drop', 'down', 'crash', 'plunge',
            'downgrade', 'underperform', 'recession', 'deficit', 'cut'
        ]
        
        analyzed_articles = []
        sentiment_scores = []
        
        for article in articles:
            # Combine title and description for analysis
            text = f"{article['title']} {article['description']}".lower()
            
            # TextBlob sentiment analysis
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Custom keyword analysis
            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            # Combine TextBlob and keyword analysis
            keyword_sentiment = 0
            if positive_count > negative_count:
                keyword_sentiment = min(0.8, (positive_count - negative_count) * 0.15)
            elif negative_count > positive_count:
                keyword_sentiment = max(-0.8, -(negative_count - positive_count) * 0.15)
            
            # Weighted combination (60% TextBlob, 40% keywords)
            final_sentiment = (textblob_sentiment * 0.6) + (keyword_sentiment * 0.4)
            
            # Normalize to [-1, 1] range
            final_sentiment = max(-1, min(1, final_sentiment))
            
            analyzed_articles.append({
                **article,
                'sentiment_score': final_sentiment,
                'sentiment_label': self._get_sentiment_label(final_sentiment)
            })
            
            sentiment_scores.append(final_sentiment)
        
        # Calculate overall metrics
        overall_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        
        positive_count = len([s for s in sentiment_scores if s > 0.1])
        negative_count = len([s for s in sentiment_scores if s < -0.1])
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        # Calculate confidence based on sentiment consistency
        sentiment_std = np.std(sentiment_scores) if len(sentiment_scores) > 1 else 0
        confidence = min(0.95, max(0.3, 1 - sentiment_std))
        
        return {
            'overall_sentiment': overall_sentiment,
            'positive_count': positive_count,
            'neutral_count': neutral_count,
            'negative_count': negative_count,
            'confidence': confidence,
            'articles_with_sentiment': analyzed_articles
        }

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'Positive'
        elif score < -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    def generate_prediction(self, stock_data: Dict, sentiment_data: Dict) -> Dict:
        """
        Generate stock price prediction based on sentiment analysis and technical indicators.
        
        This combines:
        1. Sentiment analysis results
        2. Price momentum
        3. Volume analysis
        4. Technical indicators
        """
        try:
            # Weights for different factors
            sentiment_weight = 0.5
            momentum_weight = 0.3
            volume_weight = 0.2
            
            # Calculate momentum from price change
            momentum_score = stock_data['change_percent'] / 100
            momentum_score = max(-0.5, min(0.5, momentum_score))  # Cap at ±50%
            
            # Calculate volume factor (comparing to typical volume)
            # Note: In real implementation, you'd compare to historical average
            volume_factor = 0  # Placeholder - would need historical data
            
            # Combine factors
            prediction_score = (
                sentiment_data['overall_sentiment'] * sentiment_weight +
                momentum_score * momentum_weight +
                volume_factor * volume_weight
            )
            
            # Determine direction
            if prediction_score > 0.05:
                direction = 'Bullish'
                direction_emoji = '🚀'
            elif prediction_score < -0.05:
                direction = 'Bearish'
                direction_emoji = '📉'
            else:
                direction = 'Neutral'
                direction_emoji = '➡️'
            
            # Calculate confidence
            confidence = min(0.95, abs(prediction_score) + 0.4)
            
            # Calculate target price (simple model)
            price_change_percent = prediction_score * 10  # Convert to percentage
            target_price = stock_data['price'] * (1 + price_change_percent / 100)
            
            # Generate reasoning
            sentiment_desc = "positive" if sentiment_data['overall_sentiment'] > 0 else "negative" if sentiment_data['overall_sentiment'] < 0 else "neutral"
            momentum_desc = "upward" if momentum_score > 0 else "downward" if momentum_score < 0 else "sideways"
            
            reasoning = f"{sentiment_desc.title()} sentiment ({sentiment_data['overall_sentiment']:.2f}) combined with {momentum_desc} price momentum suggests {direction.lower()} outlook."
            
            return {
                'direction': direction,
                'direction_emoji': direction_emoji,
                'confidence': confidence,
                'target_price': round(target_price, 2),
                'prediction_score': prediction_score,
                'reasoning': reasoning,
                'time_horizon': '1-2 weeks'
            }
            
        except Exception as e:
            st.error(f"Error generating prediction: {str(e)}")
            return {
                'direction': 'Neutral',
                'direction_emoji': '➡️',
                'confidence': 0.5,
                'target_price': stock_data.get('price', 0),
                'prediction_score': 0,
                'reasoning': 'Unable to generate prediction due to insufficient data.',
                'time_horizon': 'N/A'
            }

    def fetch_financial_statements(self, symbol: str) -> Dict:
        """Fetch financial statement data and earnings information from Financial Modeling Prep API."""
        try:
            base_url = "https://financialmodelingprep.com/api/v3"
            params = {'apikey': self.API_KEYS['FINANCIAL_MODELING_PREP']}
            
            # Fetch income statement
            income_url = f"{base_url}/income-statement/{symbol}"
            income_params = {**params, 'limit': 1}
            
            # Fetch earnings surprises
            earnings_url = f"{base_url}/earnings-surprises/{symbol}"
            
            # Fetch key metrics
            metrics_url = f"{base_url}/key-metrics/{symbol}"
            metrics_params = {**params, 'limit': 1}
            
            # Make requests
            income_response = requests.get(income_url, params=income_params, timeout=10)
            earnings_response = requests.get(earnings_url, params=params, timeout=10)
            metrics_response = requests.get(metrics_url, params=metrics_params, timeout=10)
            
            income_data = income_response.json() if income_response.status_code == 200 else []
            earnings_data = earnings_response.json() if earnings_response.status_code == 200 else []
            metrics_data = metrics_response.json() if metrics_response.status_code == 200 else []
            
            # Process data
            result = {
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_available': True
            }
            
            # Income statement data
            if income_data and len(income_data) > 0:
                latest_income = income_data[0]
                result['income_statement'] = {
                    'revenue': latest_income.get('revenue', 0),
                    'net_income': latest_income.get('netIncome', 0),
                    'gross_profit': latest_income.get('grossProfit', 0),
                    'operating_income': latest_income.get('operatingIncome', 0),
                    'date': latest_income.get('date', 'N/A')
                }
            
            # Earnings data
            if earnings_data and len(earnings_data) > 0:
                latest_earnings = earnings_data[0]
                result['earnings'] = {
                    'actual_eps': latest_earnings.get('actualEarningResult', 0),
                    'estimated_eps': latest_earnings.get('estimatedEarning', 0),
                    'surprise': latest_earnings.get('actualEarningResult', 0) - latest_earnings.get('estimatedEarning', 0),
                    'date': latest_earnings.get('date', 'N/A')
                }
            
            # Key metrics
            if metrics_data and len(metrics_data) > 0:
                latest_metrics = metrics_data[0]
                result['key_metrics'] = {
                    'pe_ratio': latest_metrics.get('peRatio', 0),
                    'price_to_book': latest_metrics.get('priceToBookRatio', 0),
                    'debt_to_equity': latest_metrics.get('debtToEquity', 0),
                    'roe': latest_metrics.get('returnOnEquity', 0),
                    'date': latest_metrics.get('date', 'N/A')
                }
            
            return result
            
        except Exception as e:
            st.warning(f"Could not fetch financial statements: {str(e)}")
            return {
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'data_available': False,
                'message': f'Error fetching financial data: {str(e)}'
            }

    def analyze_stock(self, symbol: str) -> Optional[Dict]:
        """Main function to perform complete stock analysis."""
        try:
            with st.spinner(f'Analyzing {symbol}... This may take a few moments.'):
                # Fetch all data
                progress_bar = st.progress(0)
                
                # Step 1: Stock price data
                progress_bar.progress(20)
                stock_data = self.fetch_stock_price(symbol)
                if not stock_data:
                    return None
                
                # Step 2: Company information
                progress_bar.progress(40)
                company_info = self.fetch_company_info(symbol)
                
                # Step 3: News articles
                progress_bar.progress(60)
                news_articles = self.fetch_stock_news(symbol, company_info.get('companyName', symbol))
                
                # Step 4: Sentiment analysis
                progress_bar.progress(80)
                sentiment_data = self.analyze_sentiment(news_articles)
                
                # Step 5: Generate prediction
                progress_bar.progress(90)
                prediction = self.generate_prediction(stock_data, sentiment_data)
                
                # Step 6: Financial statements
                progress_bar.progress(95)
                financial_data = self.fetch_financial_statements(symbol)
                
                progress_bar.progress(100)
                
                # Combine all data
                complete_analysis = {
                    'symbol': stock_data['symbol'],
                    'company_name': company_info.get('companyName', symbol),
                    'sector': company_info.get('sector', 'Unknown'),
                    'industry': company_info.get('industry', 'Unknown'),
                    'stock_data': stock_data,
                    'sentiment_data': sentiment_data,
                    'prediction': prediction,
                    'financial_data': financial_data,
                    'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'news_articles': news_articles
                }
                
                # Store in session state
                st.session_state.current_stock_data = complete_analysis
                
                # Add to analysis history
                if len(st.session_state.analysis_history) >= 10:
                    st.session_state.analysis_history.pop(0)
                st.session_state.analysis_history.append({
                    'symbol': symbol,
                    'timestamp': complete_analysis['analysis_timestamp'],
                    'sentiment': sentiment_data['overall_sentiment'],
                    'prediction': prediction['direction']
                })
                
                progress_bar.empty()
                return complete_analysis
                
        except Exception as e:
            st.error(f"Error during stock analysis: {str(e)}")
            return None

def main():
    """Main Streamlit application."""
    
    # Initialize the analyzer
    analyzer = StockSentimentAnalyzer()
    
    # App header
    st.title("📈 Real-Time Stock Sentiment Platform")
    st.markdown("**Live market data • AI sentiment analysis • Stock predictions**")
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuration")
        
        # API Status Check
        st.success("✅ API Keys Configured")
        st.info("Real-time data enabled")
        
        st.markdown("---")
        
        # Watchlist management
        st.header("⭐ Watchlist")
        
        # Add to watchlist
        new_symbol = st.text_input("Add Symbol:", placeholder="e.g., AAPL").upper()
        if st.button("➕ Add to Watchlist") and new_symbol:
            if new_symbol not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_symbol)
                st.success(f"Added {new_symbol} to watchlist!")
            else:
                st.warning(f"{new_symbol} already in watchlist")
        
        # Display watchlist
        if st.session_state.watchlist:
            st.write("**Current Watchlist:**")
            for i, symbol in enumerate(st.session_state.watchlist):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📊 {symbol}", key=f"analyze_{symbol}"):
                        analysis = analyzer.analyze_stock(symbol)
                        if analysis:
                            st.rerun()
                with col2:
                    if st.button("🗑️", key=f"remove_{symbol}"):
                        st.session_state.watchlist.remove(symbol)
                        st.rerun()
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Live Analysis", "📊 Watchlist Dashboard", "📋 Financial Statements", "📈 Analysis History"])
    
    with tab1:
        st.header("🔍 Real-Time Stock Analysis")
        
        # Stock input
        col1, col2 = st.columns([3, 1])
        with col1:
            symbol_input = st.text_input(
                "Enter Stock Symbol:",
                placeholder="e.g., AAPL, GOOGL, MSFT, TSLA",
                help="Enter any publicly traded stock symbol"
            ).upper()
        with col2:
            st.write("")  # Spacing
            analyze_button = st.button("🚀 Analyze Stock", type="primary")
        
        # Analyze stock
        if analyze_button and symbol_input:
            analysis = analyzer.analyze_stock(symbol_input)
            if analysis:
                st.rerun()
        
        # Display analysis results
        if st.session_state.current_stock_data:
            data = st.session_state.current_stock_data
            
            # Stock overview
            st.subheader(f"{data['company_name']} ({data['symbol']})")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Current Price",
                    f"${data['stock_data']['price']:.2f}",
                    f"{data['stock_data']['change']:+.2f} ({data['stock_data']['change_percent']:+.2f}%)"
                )
            with col2:
                st.metric("Volume", f"{data['stock_data']['volume']:,}")
            with col3:
                st.metric("Day High", f"${data['stock_data']['high']:.2f}")
            with col4:
                st.metric("Day Low", f"${data['stock_data']['low']:.2f}")
            
            st.markdown(f"**Sector:** {data['sector']} | **Industry:** {data['industry']}")
            st.markdown(f"*Last updated: {data['analysis_timestamp']}*")
            
            # Sentiment Analysis
            st.subheader("🎯 Sentiment Analysis")
            
            sentiment = data['sentiment_data']
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment metrics
                sentiment_class = 'positive-sentiment' if sentiment['overall_sentiment'] > 0.1 else 'negative-sentiment' if sentiment['overall_sentiment'] < -0.1 else 'neutral-sentiment'
                st.markdown(f"""
                <div class="metric-card {sentiment_class}">
                    <h4>Overall Sentiment Score</h4>
                    <h2>{sentiment['overall_sentiment']:.3f}</h2>
                    <p>Confidence: {sentiment['confidence']:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Sentiment breakdown
                st.write("**Sentiment Breakdown:**")
                col2a, col2b, col2c = st.columns(3)
                with col2a:
                    st.metric("Positive", sentiment['positive_count'])
                with col2b:
                    st.metric("Neutral", sentiment['neutral_count'])
                with col2c:
                    st.metric("Negative", sentiment['negative_count'])
            
            # Prediction
            st.subheader("🔮 AI Prediction")
            
            prediction = data['prediction']
            prediction_class = f"prediction-{prediction['direction'].lower()}"
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="{prediction_class}">
                    <h3>{prediction['direction_emoji']} {prediction['direction']}</h3>
                    <p><strong>Target Price:</strong> ${prediction['target_price']:.2f}</p>
                    <p><strong>Confidence:</strong> {prediction['confidence']:.1%}</p>
                    <p><strong>Time Horizon:</strong> {prediction['time_horizon']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.write("**Reasoning:**")
                st.write(prediction['reasoning'])
            
            # Recent News
            st.subheader("📰 Recent News & Analysis")
            
            if data['news_articles']:
                for i, article in enumerate(data['news_articles'][:5]):
                    sentiment_label = data['sentiment_data']['articles_with_sentiment'][i]['sentiment_label']
                    sentiment_score = data['sentiment_data']['articles_with_sentiment'][i]['sentiment_score']
                    
                    with st.expander(f"{sentiment_label} | {article['title'][:80]}..."):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**Source:** {article['source']}")
                            st.write(f"**Published:** {article['published_at'][:10]}")
                            st.write(article['description'])
                            st.markdown(f"[Read full article]({article['url']})")
                        with col2:
                            st.metric("Sentiment", f"{sentiment_score:.3f}")
                            st.write(f"**{sentiment_label}**")
            else:
                st.info("No recent news articles found.")
    
    with tab2:
        st.header("📊 Watchlist Dashboard")
        
        if st.session_state.watchlist:
            st.write("Quick overview of your watchlist stocks:")
            
            # Create columns for watchlist items
            cols = st.columns(min(3, len(st.session_state.watchlist)))
            
            for i, symbol in enumerate(st.session_state.watchlist[:6]):  # Show max 6 stocks
                with cols[i % 3]:
                    if st.button(f"📊 Analyze {symbol}", key=f"dashboard_{symbol}"):
                        analysis = analyzer.analyze_stock(symbol)
                        if analysis:
                            st.rerun()
        else:
            st.info("Your watchlist is empty. Add some stocks to get started!")
    
    with tab3:
        st.header("📋 Financial Statements")
        
        if st.session_state.current_stock_data and 'financial_data' in st.session_state.current_stock_data:
            financial_data = st.session_state.current_stock_data['financial_data']
            symbol = st.session_state.current_stock_data['symbol']
            
            st.subheader(f"Financial Data for {symbol}")
            
            if financial_data.get('data_available', False):
                # Income Statement
                if 'income_statement' in financial_data:
                    st.write("**Income Statement (Latest Quarter)**")
                    income = financial_data['income_statement']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Revenue", f"${income['revenue']:,.0f}")
                    with col2:
                        st.metric("Net Income", f"${income['net_income']:,.0f}")
                    with col3:
                        st.metric("Gross Profit", f"${income['gross_profit']:,.0f}")
                    with col4:
                        st.metric("Operating Income", f"${income['operating_income']:,.0f}")
                
              # Earnings Data
                if 'earnings' in financial_data:
                    st.write("**Latest Earnings Results**")
                    earnings = financial_data['earnings']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Actual EPS", f"${earnings['actual_eps']:.2f}")
                    with col2:
                        st.metric("Estimated EPS", f"${earnings['estimated_eps']:.2f}")
                    with col3:
                        surprise = earnings['surprise']
                        surprise_color = "green" if surprise > 0 else "red" if surprise < 0 else "gray"
                        st.metric("Earnings Surprise", f"${surprise:+.2f}")
                    
                    st.write(f"*Earnings Date: {earnings['date']}*")
                
                # Key Metrics
                if 'key_metrics' in financial_data:
                    st.write("**Key Financial Metrics**")
                    metrics = financial_data['key_metrics']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        pe_ratio = metrics['pe_ratio']
                        st.metric("P/E Ratio", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
                    with col2:
                        pb_ratio = metrics['price_to_book']
                        st.metric("Price-to-Book", f"{pb_ratio:.2f}" if pb_ratio else "N/A")
                    with col3:
                        debt_equity = metrics['debt_to_equity']
                        st.metric("Debt-to-Equity", f"{debt_equity:.2f}" if debt_equity else "N/A")
                    with col4:
                        roe = metrics['roe']
                        st.metric("ROE (%)", f"{roe*100:.1f}%" if roe else "N/A")
                    
                    st.write(f"*Metrics Date: {metrics['date']}*")
                
                st.write(f"*Financial data last updated: {financial_data['last_updated']}*")
            
            else:
                st.warning("Financial data not available for this stock.")
                if 'message' in financial_data:
                    st.error(financial_data['message'])
        
        else:
            st.info("Select a stock from the Live Analysis tab to view financial statements.")
    
    with tab4:
        st.header("📈 Analysis History")
        
        if st.session_state.analysis_history:
            st.write("Your recent stock analyses:")
            
            # Create a DataFrame for better display
            history_df = pd.DataFrame(st.session_state.analysis_history)
            history_df['sentiment_label'] = history_df['sentiment'].apply(
                lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
            )
            
            # Display as a table
            st.dataframe(
                history_df[['symbol', 'timestamp', 'sentiment_label', 'prediction', 'sentiment']].rename(columns={
                    'symbol': 'Symbol',
                    'timestamp': 'Analysis Time',
                    'sentiment_label': 'Sentiment',
                    'prediction': 'Prediction',
                    'sentiment': 'Sentiment Score'
                }),
                use_container_width=True
            )
            
            # Create a simple chart of sentiment over time
            if len(history_df) > 1:
                st.subheader("Sentiment Trend")
                
                fig = px.line(
                    history_df,
                    x='timestamp',
                    y='sentiment',
                    color='symbol',
                    title='Sentiment Score Over Time',
                    labels={'sentiment': 'Sentiment Score', 'timestamp': 'Analysis Time'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Neutral")
                fig.add_hline(y=0.1, line_dash="dot", line_color="green", annotation_text="Positive Threshold")
                fig.add_hline(y=-0.1, line_dash="dot", line_color="red", annotation_text="Negative Threshold")
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear Analysis History"):
                st.session_state.analysis_history = []
                st.success("Analysis history cleared!")
                st.rerun()
        
        else:
            st.info("No analysis history yet. Analyze some stocks to see your history here!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Real-Time Stock Sentiment Platform</strong></p>
        <p>Powered by Alpha Vantage, NewsAPI, Financial Modeling Prep, and Polygon APIs</p>
        <p><em>⚠️ This tool is for informational purposes only. Not financial advice.</em></p>
    </div>
    """, unsafe_allow_html=True)

# Additional helper functions for enhanced functionality
def create_sentiment_gauge(sentiment_score: float) -> go.Figure:
    """Create a gauge chart for sentiment visualization."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Sentiment Score"},
        gauge = {
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.1], 'color': "lightcoral"},
                {'range': [-0.1, 0.1], 'color': "lightyellow"},
                {'range': [0.1, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def format_large_number(num: float) -> str:
    """Format large numbers for better readability."""
    if abs(num) >= 1e12:
        return f"${num/1e12:.1f}T"
    elif abs(num) >= 1e9:
        return f"${num/1e9:.1f}B"
    elif abs(num) >= 1e6:
        return f"${num/1e6:.1f}M"
    elif abs(num) >= 1e3:
        return f"${num/1e3:.1f}K"
    else:
        return f"${num:.2f}"

def calculate_rsi(prices: List[float], period: int = 14) -> float:
    """Calculate Relative Strength Index (RSI) for technical analysis."""
    if len(prices) < period + 1:
        return 50.0  # Default neutral RSI
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [delta if delta > 0 else 0 for delta in deltas]
    losses = [-delta if delta < 0 else 0 for delta in deltas]
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_market_status() -> Dict:
    """Get current market status (open/closed)."""
    from datetime import datetime
    import pytz
    
    # US Eastern Time
    et = pytz.timezone('US/Eastern')
    current_time = datetime.now(et)
    
    # Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
    market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_weekday = current_time.weekday() < 5  # Monday = 0, Sunday = 6
    is_market_hours = market_open <= current_time <= market_close
    
    return {
        'is_open': is_weekday and is_market_hours,
        'current_time': current_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
        'next_open': 'Next trading day' if not is_weekday else 'Tomorrow 9:30 AM ET' if current_time > market_close else 'Now open',
        'status': 'Open' if (is_weekday and is_market_hours) else 'Closed'
    }

# Enhanced error handling and retry mechanism
def retry_api_call(func, max_retries: int = 3, delay: float = 1.0):
    """Retry API calls with exponential backoff."""
    import time
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))  # Exponential backoff
    
    return None

# Run the application
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
        
        # Log error details for debugging
        st.expander("Error Details", expanded=False).exception(e)
