import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

file_path = "dataset/metadata.csv"
data = pd.read_csv(file_path)
avg_token_count = 0
all_tokens = []
for text in data["text"]:
    tokens = text.split(", ")
    all_tokens.extend(tokens)
    avg_token_count += len(tokens)

print(f"Average tokens per prompt: {avg_token_count / 50}")

token_counts = Counter(all_tokens)

sorted_tokens = token_counts.most_common(20) 

wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color="white",
    colormap="viridis"
).generate_from_frequencies(token_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()