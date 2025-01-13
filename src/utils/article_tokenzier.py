import re
import json
from collections import Counter
from collections.abc import Iterable

class ArticleTokenizer:
    def __init__(self, file=None, articles=None, min_freq=0, unk_article="[UNK]", reserved_articles=None):
        self.pattern = re.compile(r'《(.+?)》第(.+?)条')

        self._article_freqs = None
        self.idx_to_article = []
        self.article_to_idx = {}
        self.idx_to_source = []
        self.article_to_source = {}
        self.article_freqs = {}

        if file is not None:
            self.load(file)
        else:
            self.unk_article = unk_article
            if articles is None:
                articles = []

            if reserved_articles is None:
                reserved_articles = []

            counter = self.count_articles(articles)
            self._article_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
            self.idx_to_article = [unk_article] + reserved_articles
            self.article_to_idx = {
                article: idx for idx, article in enumerate(self.idx_to_article)
            }

            for article, freq in self._article_freqs:
                if freq < min_freq:
                    break
                if article not in self.article_to_idx:
                    self.idx_to_article.append(article)
                    self.article_to_idx[article] = len(self.idx_to_article) - 1
            self.article_freqs = {key: value for key, value in self._article_freqs}
            self.load_source(articles)

        self.freqs = [self.article_freqs.get(article, 1) for article in self.idx_to_article]
        self.avg_freq = sum(self.freqs) / len(self)
        
    def __len__(self):
        return len(self.idx_to_article)
    
    def __call__(self, articles):
        return {"article_ids": self[articles]}
    
    def __getitem__(self, articles):
        if isinstance(articles, str):
            return self.article_to_idx.get(self.extract(articles), self.unk)
        elif isinstance(articles, Iterable):
            return [self.__getitem__(article) for article in articles]
        else:
            raise NotImplementedError

    def extract(self, article):
        match = re.search(self.pattern, article)
        if match:
            name, number = match.groups()
            return f"《{name}》第{number}条"
        return self.unk_article
    
    def save_vocab(self, vocab_file):
        with open(vocab_file, 'w', encoding='utf-8') as f:
            for article, idx in self.article_to_idx.items():
                f.write(f"{idx} {article} {self.article_freqs.get(article, 0)}\n")

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                idx, article, freq = line.strip().split()
                self.article_to_idx[article] = int(idx)
                self.article_freqs[article] = int(freq)
                self.idx_to_article.append(article)
        self.unk_article = self.idx_to_article[0]  # 假设第一个条目是未知文章的标记
    
    def count_articles(self, articles):
        tokens = [self.extract(article) for article in articles]
        return Counter(tokens)
    
    def load_source(self, articles):
        self.idx_to_source = [None] * len(self)
        for article in articles:
            item = self.extract(article)
            idx = self.article_to_idx[item]
            if self.idx_to_source[idx] is None:
                self.idx_to_source[idx] = article
        for idx in range(len(self)):
            self.article_to_source[self.idx_to_article[idx]] = self.idx_to_source[idx]

    
    def convert_article_to_source(self, article):
        if isinstance(article, str):
            return self.article_to_source.get(article, article)
        elif isinstance(article, Iterable):
            return [self.convert_article_to_source(a) for a in article]
        else:
            raise NotImplementedError
        
    def convert_idx_to_source(self, idx):
        if not isinstance(idx, Iterable):
            return self.idx_to_source[idx]
        return [self.idx_to_source[id] for id in idx]
    
    def convert_idx_to_article(self, idx):
        if not isinstance(idx, Iterable):
            return self.idx_to_article[idx]
        return [self.idx_to_article[id] for id in idx]
    
    def convert_article_to_idx(self, article):
        return self[article]
    
    def save(self, file_name):
        json_data = []
        for idx, article, source in zip(range(len(self)), self.idx_to_article, self.idx_to_source):
            json_data.append({'idx': idx, 'article': article, 'source text': source, 'freq': self.article_freqs.get(article, 0)})

        with open(file_name, 'w') as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

    def load(self, file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)

        for item in data:
            idx, article, source, freq = item["idx"], item["article"], item["source text"], item["freq"]
            self.idx_to_article.append(article)
            self.idx_to_source.append(source)
            self.article_to_idx[article] = idx
            self.article_to_source[article] = source
            self.article_freqs[article] = freq
        self.unk_article = self.idx_to_article[0]

    @property
    def unk(self):
        return 0
                

# 使用示例
