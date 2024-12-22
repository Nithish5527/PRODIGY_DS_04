import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Example synthetic dataset: Social media posts
data = {
    "Post": [
        "I absolutely love this product! It's amazing.",
        "The service was terrible, never using it again.",
        "Pretty decent overall, could be better.",
        "I have no strong feelings about this.",
        "Absolutely fantastic experience, highly recommend!",
        "Worst decision ever, I'm so disappointed.",
        "It's okay, nothing special but not bad either.",
        "Amazing quality and great customer service.",
        "Horrible product, waste of money.",
        "Good value for the price, satisfied with the purchase."
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Sentiment analysis using TextBlob
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Polarity: -1 (negative) to 1 (positive)

# Apply sentiment analysis to each post
df["Sentiment"] = df["Post"].apply(analyze_sentiment)

# Categorize sentiments
def categorize_sentiment(polarity):
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment Category"] = df["Sentiment"].apply(categorize_sentiment)

# Count sentiment categories
sentiment_counts = df["Sentiment Category"].value_counts()

# ----------- Visualization ----------- #
# Bar chart for sentiment categories
plt.figure(figsize=(8, 5))
plt.bar(sentiment_counts.index, sentiment_counts.values, color=["green", "red", "gray"])

# Customize the plot
plt.title("Sentiment Distribution", fontsize=16)
plt.xlabel("Sentiment", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.tight_layout()

# Show the bar chart
plt.show()

# Line chart for sentiment polarity
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["Sentiment"], marker="o", linestyle="-", color="blue")

# Customize the plot
plt.title("Sentiment Polarity Over Posts", fontsize=16)
plt.xlabel("Post Index", fontsize=14)
plt.ylabel("Sentiment Polarity", fontsize=14)
plt.axhline(0, color="gray", linestyle="--")  # Neutral line
plt.tight_layout()

# Show the line chart
plt.show()

# Display the DataFrame with sentiments
print(df)
