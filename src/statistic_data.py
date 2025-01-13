from utils.article_tokenzier import ArticleTokenizer
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

# 设置Seaborn的样式，关闭网格线
sns.set(context="paper", style="whitegrid", palette="deep")

article_vocab_file="tem.json"
article_tokenizer = ArticleTokenizer(file=article_vocab_file)

freqs = article_tokenizer.freqs

freqs = [freq for freq in freqs if freq > 1]
ids   = list(range(len(freqs)))

data = {
    'ids': ids,
    'freqs': freqs,
}

df = pd.DataFrame(data)


# 创建图形和brokenaxes实例
fig = plt.figure(figsize=(10, 6))  # 设置图形大小

sns.histplot(data=df, x="freqs")


plt.savefig('statistic.pdf', bbox_inches='tight', dpi=1000)

# 显示图形
plt.show()