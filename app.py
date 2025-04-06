from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import nltk
nltk.download('vader_lexicon')

# Initialize the Flask application
app = Flask(__name__)

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()



# Route for the home page
@app.route("/", methods=["GET"])
def index():
    # Render the input form page
    return render_template("index.html")

# Route to handle the form submission and analysis
@app.route("/analyze", methods=["POST"])
def analyze():
    # Get the URL from the form
    url = request.form.get("url")
    
    try:
        # Fetch the webpage using requests
        response = requests.get(url)
        if response.status_code != 200:
            return f"Error: Unable to retrieve the page (status code {response.status_code}).", 400
    except Exception as e:
        return f"Error: {e}", 500

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract all reviews assuming they are in tags with class 'review-text'
    review_elements = soup.find_all(class_="review-text")
    
    # If no reviews found, return an error message.
    if not review_elements:
        return "Error: No reviews found on this page. Check the HTML element selector.", 400

    # List to hold review texts and a dictionary to hold their sentiment scores.
    reviews = []
    sentiment_scores = []
    
    # Loop through each extracted element and get its text
    for elem in review_elements:
        review_text = elem.get_text(strip=True)
        if review_text:
            reviews.append(review_text)
            # Get sentiment score using VADER
            scores = sia.polarity_scores(review_text)
            sentiment_scores.append(scores)
    
    total_reviews = len(reviews)
    
    # Separate reviews into positive (pros) and negative (cons) based on compound score thresholds
    pros = []
    cons = []
    # We'll also keep track of the highest positive and lowest negative for the example.
    best_positive = None
    best_positive_score = -1.0  # start low
    worst_negative = None
    worst_negative_score = 1.0   # start high

    for review, scores in zip(reviews, sentiment_scores):
        compound = scores["compound"]
        if compound >= 0.05:
            pros.append(review)
            if compound > best_positive_score:
                best_positive_score = compound
                best_positive = review
        elif compound <= -0.05:
            cons.append(review)
            if compound < worst_negative_score:
                worst_negative_score = compound
                worst_negative = review

    pos_count = len(pros)
    neg_count = len(cons)

    # If there are no positive or negative reviews, provide fallback messages.
    if not best_positive:
        best_positive = "No clearly positive review found."
    if not worst_negative:
        worst_negative = "No clearly negative review found."

    # Render the result page using the result template
    return render_template("results.html",
                                  total_reviews=total_reviews,
                                  pos_count=pos_count,
                                  neg_count=neg_count,
                                  pros=pros,
                                  cons=cons,
                                  example_pro=best_positive,
                                  example_con=worst_negative)

# Run the app in debug mode
if __name__ == "__main__":
    app.run(debug=True)