# Viva Questions - Text Classification (Bag-of-Words, Tokenize & Vectorize)

### 1. What is the objective of your experiment?
**Answer:**  
The objective is to implement and study a Bag-of-Words (BoW) text-classification pipeline: tokenization, preprocessing, vectorization (Count/TF-IDF/Hashing), model training (Naive Bayes, Logistic Regression, SVM), evaluation and error analysis. The experiment demonstrates feature extraction choices, weighting schemes, and practical issues like sparsity, smoothing, and OOV handling.

---

### 2. What is the Bag-of-Words model?
**Answer:**  
BoW is a representation that maps a document to a fixed-length vector of token counts or weights from a vocabulary. It ignores word order and syntax; each dimension corresponds to a token (or n-gram) and the value is typically raw count, binary presence, or TF-IDF weight.

---

### 3. What preprocessing steps are important before vectorizing?
**Answer:**  
Common steps: lowercasing, removing or normalizing punctuation, unicode normalization, removing/handling stopwords, expanding contractions, tokenization, optional stemming or lemmatization, and mapping special tokens (URLs, numbers) to placeholders. Choice depends on task — aggressive cleaning can remove informative signals.

---

### 4. Explain TF, IDF and TF-IDF.
**Answer:**  
TF (term frequency) counts term occurrences in a document. IDF (inverse document frequency) downweights terms that appear in many documents: idf(t)=log((N+1)/(df_t+1))+1 (smoothed). TF-IDF = TF × IDF highlights terms that are frequent in a document but rare across the corpus, improving discriminative power.

---

### 5. Compare CountVectorizer, TfidfVectorizer and HashingVectorizer.
**Answer:**  
CountVectorizer produces raw counts and stores a vocabulary mapping. TfidfVectorizer computes TF-IDF weights and also stores vocabulary. HashingVectorizer applies a hash function to map tokens to fixed-size indices without storing vocabulary — memory efficient and streaming-friendly but susceptible to collisions and not invertible.

---

### 6. What is the hashing trick and when to use it?
**Answer:**  
The hashing trick maps tokens to indices via a hash function into a fixed-size vector. It's useful for large-scale or streaming pipelines where storing a vocabulary is costly, or when you want fixed memory usage. Downsides: collisions and inability to retrieve original tokens from indices.

---

### 7. Which models work well with BoW features?
**Answer:**  
Linear models and probabilistic models perform well: Multinomial/Bernoulli Naive Bayes (fast on count data), Logistic Regression (with L1/L2 regularization), and linear SVMs. Tree-based models are less common on very high-dimensional sparse BoW unless dimensionality reduction is applied.

---

### 8. How does Multinomial Naive Bayes compute class probability for a document?
**Answer:**  
For class c and document d with term counts x_t: P(c|d) ∝ P(c) × ∏_t P(t|c)^{x_t}. With Laplace smoothing: P(t|c) = (count(t,c) + α) / (total_terms_c + αV). Compute log-probabilities to avoid underflow: log P(c|d) = log P(c) + Σ_t x_t log P(t|c).

---

### 9. Why use Laplace smoothing?
**Answer:**  
To avoid zero probabilities for unseen term-class pairs. Add-α smoothing ensures every possible token has non-zero probability, preventing log(0) during inference and providing more stable estimates for rare events.

---

### 10. How do you handle out-of-vocabulary (OOV) words at inference?
**Answer:**  
Options: ignore them (Count/TF-IDF vectorizer will not map OOV tokens), map them to an `UNK` token during preprocessing, or use HashingVectorizer which implicitly maps new tokens into the fixed-size vector (collisions possible). For probabilistic models, ensure smoothing provides non-zero emission for unseen tokens.

---

### 11. How do you evaluate a text-classification model?
**Answer:**  
Use accuracy, precision, recall, F1-score (micro/macro/weighted), and confusion matrix. For imbalanced classes prefer precision/recall or the precision-recall curve and class-weighted metrics. Use stratified k-fold CV for reliable estimates.

---

### 12. How do you select hyperparameters for BoW pipelines?
**Answer:**  
Use GridSearchCV or RandomizedSearchCV to tune ngram_range, `min_df`/`max_df`, C (regularization for LR/SVM), α (for NB), and vectorizer choices. Employ nested CV for unbiased hyperparameter estimates when data is limited.

---

### 13. How to reduce dimensionality or feature set for BoW?
**Answer:**  
Approaches: statistical feature selection (chi-square, mutual information, SelectKBest), use `min_df`/`max_df` thresholds, L1 regularization for sparse feature selection, latent semantic analysis (SVD/LSA) for dense reduced features, or random projection.

---

### 14. Explain class imbalance handling techniques.
**Answer:**  
Resampling (oversample minority, undersample majority), use class-weighted losses in models (e.g., `class_weight='balanced'` in sklearn), threshold tuning, and careful selection of evaluation metrics that reflect minority-class performance.

---

### 15. Give a short scikit-learn pipeline example for BoW classification.
**Answer:**  
Conceptual example:  
```
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
  ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=5)),
  ('clf', LogisticRegression(C=1.0, class_weight='balanced'))
])
pipe.fit(X_train, y_train)
```  
Use `GridSearchCV` to tune `ngram_range`, `min_df`, and `C`.

---

### 16. How to persist and deploy vectorizers and models?
**Answer:**  
Serialize the fitted vectorizer and model using `joblib.dump()` or `pickle`. At inference load both artifacts and call `vectorizer.transform()` followed by `model.predict()`. For APIs, ensure consistent preprocessing pipeline is applied in production.

---

### 17. What are common failure modes and debugging tips?
**Answer:**  
- Data leakage (features derived from test labels).  
- Imbalanced classes causing biased predictions.  
- OOV tokens at inference time; check vocabulary coverage.  
- Overfitting on training data — use regularization and CV.  
- Inspect feature weights (linear models) and confusion matrix to identify problematic classes.

---

### 18. When is it better to use embeddings (word2vec/BERT) than BoW?
**Answer:**  
Use embeddings when context, semantics, polysemy or syntactic structure matters (sentiment with negation, semantic similarity, contextual classification). Embeddings capture semantic relationships and generalize better for rare words; BoW is a strong baseline for many classical classification problems and is faster and simpler.

---

### 19. Practical tips to scale BoW pipelines to large datasets?
**Answer:**  
Use `HashingVectorizer` for fixed memory footprint, incremental learning algorithms (e.g., `SGDClassifier` with partial_fit), process data in batches, use sparse matrices (CSR), and apply feature selection to reduce dimensionality.

---

### 20. Ethical considerations when deploying text classifiers?
**Answer:**  
Watch for biased training data that amplifies stereotypes, protect user privacy for text data, measure fairness across demographic groups, and provide error handling/explanations for automated decisions. Document dataset provenance and limitations.

---

End of Viva - Experiment 2 (Text Classification: BoW)
