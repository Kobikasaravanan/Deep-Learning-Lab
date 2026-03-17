from transformers import pipeline
import matplotlib.pyplot as plt

# Pretrained Hugging Face sentiment model
classifier = pipeline("sentiment-analysis")

# Sample sentences
texts = [
    "I love this product",
    "This movie is amazing",
    "I hate this service",
    "This is the worst experience"
]

results = []

# Predict sentiment
for text in texts:
    result = classifier(text)[0]
    print(text, "->", result["label"], result["score"])
    results.append(result["score"])

# Graph Output
plt.plot(results)
plt.title("Sentiment Prediction Scores")
plt.xlabel("Sentence")
plt.ylabel("Confidence Score")
plt.show()

