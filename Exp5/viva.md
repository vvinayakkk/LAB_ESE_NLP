## Theory and Concepts

1. What is an HMM?
- A Hidden Markov Model (HMM) models sequences where hidden states (tags) emit observable tokens (words).
- Use-case: sequence labeling tasks like NER or POS tagging.

2. Core components (simple)
- Start probabilities π: P(tag at position 0).
- Transition matrix A: P(tag_t | tag_{t-1}).
- Emission matrix B: P(word | tag).

3. States vs Observations
- States = discrete labels (e.g., PER, LOC, ORG, O).
- Observations = tokens in the sentence (preferably normalized, e.g., lowercase).

4. Viterbi (high level)
- Dynamic programming that finds the most likely state sequence.
- Maintain dp[t][s] = best log-prob for state s at time t and back[t][s] = best previous state.
- Steps: initialize → recurse (max over previous states) → terminate → backtrack.

5. Evaluation & limitations (brief)
- Viterbi gives the best path, not marginal probabilities. For token marginals use forward-backward.
- HMM is first-order Markov; it ignores long-range dependencies and rich features. Consider CRF or neural models for better performance.

---

# Viva Questions - HMM & Viterbi Decoding

### 1. What is the objective of your experiment?
**Answer:**  
The objective is to implement a Hidden Markov Model (HMM) with Viterbi decoding to perform sequence labeling tasks such as Named Entity Recognition (NER) or POS tagging. In this experiment the focus is on tagging entities like `PER`, `LOC`, `ORG`, and `O` (outside). The model is trained on labeled sequences to learn:
- start probabilities (probability a tag starts a sequence),
- transition probabilities (probability of moving from one tag to another),
- emission probabilities (probability of observing a word given a tag).  
At prediction time the Viterbi algorithm uses these probabilities to compute the most likely sequence of hidden tags for an unseen sentence. The goal is to demonstrate how an HMM can model sequential dependencies between tags and how smoothing and log-space computations make inference robust.

---

### 2. What are the hidden states and observations in your HMM?
**Answer:**  
- Hidden states: the tag set used for labeling; in this experiment these are `PER`, `LOC`, `ORG`, and `O`. These are not directly observed in test data and must be inferred.  
- Observations: the actual words (tokens) in each sentence. Observations are preprocessed (converted to lowercase) so that the emission model is case-insensitive and vocabulary size is reduced, which helps generalization and reduces sparsity.

---

### 3. Explain the three main probability matrices used in HMM.
**Answer:**  
HMM uses three core probability distributions:
- Start probability π: π[tag] = P(tag at position 0). This describes how likely each tag is to begin a sequence. Estimated from counts of first-tag occurrences in training data (with smoothing).
- Transition probability A: A[prev][curr] = P(curr | prev). This matrix encodes the probability of transitioning from the previous tag to the current tag. Estimated from counts of tag bigrams (prev → curr).
- Emission probability B: B[tag][word] = P(word | tag). This encodes how likely a tag generates a given word. Estimated from counts of (word, tag) pairs.  
All these probabilities are typically estimated from relative frequencies in training data and regularized using Laplace (add-one) smoothing to handle zero counts.

---

### 4. Why do you use Laplace (add-one) smoothing?
**Answer:**  
Smoothing prevents zero probabilities for events unseen in training. Without smoothing:
- If a word never appeared with a particular tag, the emission probability P(word|tag) = 0. In log-space this becomes -∞ and breaks Viterbi decoding.  
Add-one smoothing (Laplace) adjusts counts so every possible event has at least a small non-zero probability:
P(word|tag) = (count(word, tag) + 1) / (total_words_for_tag + |V|)  
where |V| is the vocabulary size. Similarly, transition counts are smoothed:
P(curr|prev) = (count(prev→curr) + 1) / (count(prev) + S)  
where S is the number of states. Smoothing ensures robustness to rare or unseen words/transitions and makes probability estimates well-defined.

---

### 5. Explain the Viterbi algorithm step-by-step.
**Answer:**  
Viterbi finds the most probable sequence of hidden states given the observation sequence and the HMM parameters (π, A, B). Steps:

- Initialization (t = 0):  
  For each tag s:  
  V[0, s] = log(π[s]) + log(B[s][word0])  
  back[s, 0] = 0 (or marked as start)
- Recursion (t = 1..n-1):  
  For each current tag s and each time t, compute:  
  V[t, s] = max_{p in states} [ V[t-1, p] + log(A[p][s]) ] + log(B[s][word_t])  
  back[s, t] = argmax_{p in states} [ V[t-1, p] + log(A[p][s]) ]  
  This selects the best previous tag p that maximizes the log-probability of the prefix ending in state s at time t.
- Termination:  
  best_last = argmax_s V[n-1, s]  
  best_score = max_s V[n-1, s]
- Backtracking:  
  Recover the best path by tracing back pointers:  
  tags[n-1] = best_last  
  for t from n-1 down to 1: tags[t-1] = back[tags[t], t]  
This yields the most probable tag sequence under the model. Using log-probabilities turns products into sums and avoids underflow.

---

### 6. Why do you use log probabilities in Viterbi?
**Answer:**  
Multiplying many small probabilities leads to numerical underflow (results tending to zero, losing precision). By using logarithms:
- Products become sums: log(a × b × c) = log(a) + log(b) + log(c).
- Comparisons for maximization remain valid because log is monotonic.
- Addition of logs is numerically stable relative to multiplying small floating numbers.  
Thus we compute and compare log-probabilities throughout Viterbi to maintain accuracy and stability.

---

### 7. How do you handle unknown words during prediction?
**Answer:**  
Unknown words (not seen in training) will not appear in emission counts. To avoid zero emission probability, the emission function returns a small default probability when the word is not present in the emission dictionary for a tag. For example:
- In code: `return self.emit_prob[tag].get(word, 1e-6)`  
Alternatively, if using smoothed emission tables that include vocabulary |V|, unseen words receive probability (0 + 1) / (total_emissions[tag] + |V|). If an explicit fallback is used, choose a sufficiently small value (e.g., 1e-6) - but be consistent with how you compute log-probabilities to avoid biasing results.

---

### 8. What is the purpose of the back matrix in your Viterbi implementation?
**Answer:**  
The back matrix (or backpointer) stores the index (or identity) of the best previous state for each current state at each time step:
- `back[s, t]` = best previous state p that maximizes the value for state s at time t.  
During backtracking, the back matrix is used to reconstruct the optimal sequence by following pointers from the best final state backwards to the start. Without the back matrix you can compute the best final score but cannot efficiently recover the corresponding state sequence.

---

### 9. How do you compute confidence scores for each predicted tag?
**Answer:**  
Confidence per token can be derived from the normalized probabilities at that time step. Given the Viterbi scores (log-space) dp[:, t] for all states at position t, convert them to a probability distribution via the log-softmax trick:
1. m = max_j dp[j, t] (stabilize)  
2. probs_j = exp(dp[j, t] - m)  
3. normalized_probs = probs / sum(probs)  
The confidence for the chosen tag at position t is normalized_probs[tag_index]. This yields a per-token probability (in [0,1]) reflecting how probable the chosen tag is relative to alternatives. Note: Viterbi scores correspond to the best path ending in each state; they are not the marginal probabilities from the forward-backward algorithm, but normalized Viterbi scores are useful approximations of per-token confidence.

---

### 10. What is the time complexity of the Viterbi algorithm in your code?
**Answer:**  
Let S be the number of states (tags) and N be the sentence length. For each time step t (N steps) and for each current state s (S states) we scan all previous states p (S) to compute the maximum. Thus the complexity is O(N × S × S) = O(N × S²). Memory complexity is O(N × S) to store the DP table and backpointers.

---

### 11. How are transition probabilities computed in your code?
**Answer:**  
Transition probabilities are estimated from counts of consecutive tags in training sequences and smoothed with add-one smoothing. For a previous state `prev` and next state `next`:
`trans_prob[prev][next] = (count(prev→next) + 1) / (count(prev) + S)`  
where `count(prev→next)` is how many times `prev` was followed by `next` in training, `count(prev)` is the total transitions observed from `prev`, and `S` is the number of states (for smoothing). This ensures each row of the transition matrix sums to 1 and handles unseen transitions.

---

### 12. What does model.extract_entities() do?
**Answer:**  
`model.extract_entities()` converts a sequence of token-level tags (often IOB or simple tag labels) into grouped named entities. It typically:
- Iterates tokens and their predicted tags.
- Groups consecutive tokens that share the same non-`O` tag into a single entity phrase (e.g., `"New"` + `"York"` with tag `LOC` → `"New York"` as one LOC entity).
- Handles boundaries where tags change or when encountering `O`.
- Returns a structured dictionary of entities, for example: `{ "PER": ["John Doe"], "LOC": ["New York"] }`.  
This function is useful for presenting model output in an entity-level format rather than token-level tags.

---

### 13. Why do you convert words to lowercase in training and prediction?
**Answer:**  
Lowercasing reduces vocabulary size and mitigates data sparsity. Words that differ only by capitalization (`"Paris"` vs `"paris"`) would otherwise be treated as distinct tokens, reducing the effective counts for emission probabilities. Lowercasing makes the model case-insensitive, improving generalization especially when training data is limited. Note: if capitalization is itself a strong feature (e.g., initial capitalized words often indicate entities), you may want to retain a capitalization feature instead of full lowercasing.

---

### 14. What happens if a sentence starts with a word never seen in training?
**Answer:**  
If the first word is unseen:
- Emission probabilities for all tags at t=0 will be very small (either the smoothed probability based on |V| or a fallback like 1e-6).
- The start probability π still influences the initial choice; tags that commonly start sequences will be preferred.
- Viterbi will combine start and emission probabilities; typically the model will fall back to the most frequent starting tag(s) in training.
- Confidence for the chosen tag will likely be low because emissions contributed little discriminative information.

---

### 15. Can this model be used for actual POS tagging (like NOUN, VERB)?
**Answer:**  
Yes. The HMM + Viterbi architecture is a general sequence labeling framework. To use it for POS tagging:
- Replace the state set with POS tags (e.g., `NOUN`, `VERB`, `ADJ`, ...).  
- Train emission and transition probabilities on a POS-tagged corpus (e.g., the Penn Treebank).  
The algorithmic machinery (start/transition/emission estimation and Viterbi decoding) remains the same. Performance will depend on training data size and tag complexity; for state-of-the-art POS tagging, more advanced models (CRF, neural models) often perform better.

---

### 16. What are the limitations of this HMM model?
**Answer:**  
- First-order Markov assumption: each tag depends only on the previous tag; long-range dependencies are ignored.  
- Limited contextual features: emissions depend on the current word only, not on surrounding words or handcrafted features.  
- Poor handling of unseen words and rarer patterns without extensive smoothing or feature engineering.  
- No representation learning: it does not use continuous word embeddings which capture semantic similarity.  
More powerful alternatives include BiLSTM-CRF and transformer-based models (BERT-CRF) which model long-range dependencies and utilize dense representations.

---

### 17. How would you improve this model?
**Answer:**  
Possible improvements:
- Use higher-order Markov models (bigram/trigram tag dependencies) to capture more context in transitions.  
- Add word-shape and orthographic features (capitalization, digits, hyphens) as additional emission signals.  
- Employ pre-trained word embeddings (GloVe, FastText) and measure similarity between word embeddings and tag representations.  
- Replace HMM with a discriminative sequence model such as CRF or a neural architecture (BiLSTM-CRF, Transformer + CRF).  
- Increase training data and use domain adaptation techniques to handle unseen vocabulary.

---

### 18. In your heatmap, why are `O→O` transitions high?
**Answer:**  
Most tokens in typical text are non-entities and thus labeled `O`. Therefore consecutive non-entity tokens are extremely common in sequences, producing a high empirical count for `O → O` transitions. After normalization this leads to a high `P(O|O)` value. This reflects the data distribution rather than a modeling error.
