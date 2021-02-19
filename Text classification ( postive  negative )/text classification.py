from textblob import TextBlob

text = "it was a good day"

tb = TextBlob(text)
polarity = tb.sentiment.polarity

if polarity > 0:
    print("Positive Text ( + )")
elif polarity < 0:
    print("Negative Text ( - )")
elif polarity==0:
    print("Neutral Text ( 0 )")
else:
    print("not clear")


