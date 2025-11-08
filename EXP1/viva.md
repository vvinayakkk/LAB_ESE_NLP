# Viva Questions - NLTK & Basic Corpus Analysis

### 1. What is the objective of your experiment?
**Answer:**  
The objective is to install and use NLTK to perform basic corpus analysis: compute frequency distributions, analyze morphological features (stemming/lemmatization/POS), detect collocations and n-grams, and visualize term importance with word clouds. The goal is to familiarize with NLTK tooling, typical preprocessing steps, and exploratory techniques used in NLP EDA.

---

### 2. How do you install NLTK and required corpora/models?
**Answer:**  
Install NLTK via pip: `pip install nltk`. Download datasets and models using `nltk.download()` or the command line downloader: `python -m nltk.downloader punkt wordnet averaged_perceptron_tagger stopwords` (add other corpora as needed). For automated scripts, call `nltk.download('resource_name')` programmatically.

---

### 3. What is a corpus and which corpora does NLTK provide?
**Answer:**  
A corpus is a collection of texts used for analysis. NLTK provides corpora such as Gutenberg, Brown, Reuters, Penn Treebank, movie_reviews, and WordNet (lexical database). Corpus readers give convenient access to raw text, tokenized words, tagged sentences and categorized texts.

---

### 4. What tokenizers does NLTK provide and when to use them?
**Answer:**  
Common tokenizers: `word_tokenize` (Punkt/Treebank), `sent_tokenize` (Punkt sentence tokenizer), `regexp_tokenize`, and specialized tokenizers like `TreebankWordTokenizer`. Use `sent_tokenize` for sentence splitting, `word_tokenize` for general tokenization, `regexp_tokenize` for domain-specific rules, and language-specific tokenizers for non-English texts.

---

### 5. How do you compute frequency distributions and what methods are useful?
**Answer:**  
Use `nltk.FreqDist(tokens)` to compute term frequencies. Useful methods: `.most_common(n)` to list top tokens, `.plot(n)` to visualize, `.hapaxes()` for singletons, and `nltk.ConditionalFreqDist` to compute frequencies conditioned on categories (e.g., genre). Update FreqDist incrementally for large corpora.

---

### 6. What is POS tagging and why is it important for morphological analysis?
**Answer:**  
POS tagging assigns syntactic categories (e.g., NN, VB) to tokens. It is important because lemmatizers rely on POS to choose the correct base form (e.g., "better" → "good" as adjective). POS tags also enable downstream tasks like chunking, syntactic analysis, and more accurate information extraction.

---

### 7. Difference between stemming and lemmatization?
**Answer:**  
Stemming (Porter, Lancaster, Snowball) applies heuristic truncation to produce stems; it is fast but may produce non-words. Lemmatization (WordNetLemmatizer) returns dictionary lemmas using morphological rules and POS tags; it is more accurate but requires resources and POS information.

---

### 8. How do you lemmatize words correctly with NLTK?
**Answer:**  
Tokenize, POS-tag using `nltk.pos_tag()`, map NLTK POS tags to WordNet POS categories, and call `WordNetLemmatizer().lemmatize(token, pos=wn_pos)`. This yields more accurate lemmas than calling lemmatizer without POS.

---

### 9. What are collocations and how do you find them?
**Answer:**  
Collocations are frequently co-occurring word pairs/phrases (e.g., "United States"). Use `BigramCollocationFinder`/`TrigramCollocationFinder` and scoring functions like PMI (pointwise mutual information) or likelihood ratio to rank candidate collocations.

---

### 10. What is Zipf's law and how do you visualize it?
**Answer:**  
Zipf's law states that the frequency of a word is approximately proportional to 1/rank. Visualize via a log-log rank-frequency plot (log(rank) vs log(frequency)); a near-linear trend indicates Zipfian behavior.

---

### 11. How do you generate a word cloud from token frequencies?
**Answer:**  
Use the `wordcloud` Python library: build a frequency dictionary from tokens, remove stopwords and normalize tokens, then `WordCloud().generate_from_frequencies(freq_dict)` and save or display the image. Tune parameters like `max_words`, `background_color`, and `stopwords` set.

---

### 12. How do you handle special tokens (URLs, emails, emojis) during tokenization?
**Answer:**  
Use regex-based tokenization to capture URLs/emails as single tokens, or pre-replace them with canonical placeholders (`<URL>`, `<EMAIL>`). For emojis, use emoji-aware tokenizers or libraries that treat emoji as tokens; consider whether to keep or map them depending on task.

---

### 13. How to evaluate POS tagger quality?
**Answer:**  
Compare predicted tags to a gold-standard annotated corpus (e.g., Penn Treebank) and compute accuracy. Use confusion matrices to diagnose frequent mis-tags and per-tag precision/recall for deeper analysis.

---

### 14. How to process very large corpora that don't fit in memory?
**Answer:**  
Stream documents and update counters incrementally (e.g., update a `FreqDist` or `collections.Counter`), use corpus readers that yield documents lazily, or leverage disk-backed data structures and database-backed processing pipelines.

---

### 15. How can WordNet be used in analysis?
**Answer:**  
WordNet provides synsets, synonyms, antonyms, hypernyms and example sentences. Use it for query expansion, sense disambiguation, semantic similarity approximations, and to support lemmatization and synonym-based features.

---

### 16. Common pitfalls and troubleshooting tips?
**Answer:**  
- Wrong encoding: ensure UTF-8 when reading files.  
- Tokenizer mismatch: pick tokenizers suited for domain (news, tweets).  
- Overzealous stopword removal: may remove informative tokens.  
- Not validating outputs: always inspect samples after each preprocessing step.

---

### 17. Give a short example workflow (code sketch) for frequency analysis.
**Answer:**  
Conceptual steps:  
```
from nltk import word_tokenize, FreqDist
text = open('sample.txt', encoding='utf-8').read()
tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
fd = FreqDist(tokens)
fd.plot(30)
```
This tokenizes, normalizes lowercasing, filters non-alpha tokens, builds a frequency distribution and plots the top 30 words.

---

End of Viva - Experiment 1 (NLTK & Basic Corpus Analysis)
