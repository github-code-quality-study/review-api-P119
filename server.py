import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Parse query parameters
            query_params = parse_qs(environ.get('QUERY_STRING', ''))
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            # Filter reviews based on query parameters
            filtered_reviews = [
                review for review in reviews
                if (location is None or review['Location'] == location) and
                (start_date is None or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= datetime.strptime(start_date, '%Y-%m-%d')) and
                (end_date is None or datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= datetime.strptime(end_date, '%Y-%m-%d'))
            ]

            # Add sentiment analysis to each filtered review
            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            # Sort the filtered reviews by the compound value in sentiment in descending order
            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            # Create the response body from the filtered reviews and convert to a JSON byte string
            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            # Set the appropriate response headers
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            # Parse the POST request body
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except (ValueError):
                request_body_size = 0

            request_body = environ['wsgi.input'].read(request_body_size).decode('utf-8')
            post_params = parse_qs(request_body)

            review_body = post_params.get('ReviewBody', [None])[0]
            location = post_params.get('Location', [None])[0]

            # List of valid locations
            valid_locations = [
                "Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California",
                "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California",
                "El Paso, Texas", "Escondido, California", "Fresno, California",
                "La Mesa, California", "Las Vegas, Nevada", "Los Angeles, California",
                "Oceanside, California", "Phoenix, Arizona", "Sacramento, California",
                "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona"
            ]

            if review_body and location:
                if location not in valid_locations:
                    # Handle invalid location
                    error_message = {"error": "Invalid Location"}
                    response_body = json.dumps(error_message, indent=2).encode("utf-8")

                    start_response("400 Bad Request", [
                        ("Content-Type", "application/json"),
                        ("Content-Length", str(len(response_body)))
                    ])
                    
                    return [response_body]

                # Generate ReviewId and Timestamp
                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # Create a new review dictionary
                new_review = {
                    "ReviewId": review_id,
                    "ReviewBody": review_body,
                    "Location": location,
                    "Timestamp": timestamp
                }

                # Append the new review to the reviews list
                reviews.append(new_review)

                # Create the response body from the new review and convert to a JSON byte string
                response_body = json.dumps(new_review, indent=2).encode("utf-8")

                # Set the appropriate response headers
                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                
                return [response_body]
            else:
                # Handle missing parameters
                error_message = {"error": "Missing ReviewBody or Location"}
                response_body = json.dumps(error_message, indent=2).encode("utf-8")

                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                
                return [response_body]
    
if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()