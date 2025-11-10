# 🧠 Viva Questions and Answers – Named Entity Recognition (NER) using LSTM + CRF

---

### **1. What is Named Entity Recognition (NER)?**
NER is a Natural Language Processing (NLP) technique that identifies and classifies key information (entities) in text into predefined categories such as **Person (PER)**, **Location (LOC)**, **Organization (ORG)**, and **Miscellaneous (MISC)**.

---

### **2. What is the difference between NER and POS Tagging?**
- **NER** identifies *meaningful named entities* (like names, cities, companies).  
- **POS Tagging** identifies *grammatical parts of speech* (like nouns, verbs, adjectives).  
POS tagging focuses on syntax, while NER focuses on semantics.

---

### **3. Why is LSTM preferred over simple RNN for NER?**
Traditional RNNs suffer from the **vanishing gradient problem** and struggle to remember long-term dependencies.  
**LSTMs** use gates (input, forget, and output) to store and manage long-term information, making them ideal for **sequence labeling tasks** like NER where context is crucial.

---

### **4. What is a Bidirectional LSTM?**
A **Bidirectional LSTM** processes the input sequence **in both forward and backward directions**, allowing the model to use both **past and future context** when predicting entity tags.  
This improves accuracy because the meaning of a word often depends on its surrounding words.

---

### **5. What is the role of the CRF layer in this model?**
The **Conditional Random Field (CRF)** layer captures **dependencies between output labels** and ensures valid tag transitions.  
For example, it prevents invalid sequences like `I-ORG` directly following `B-PER`.  
It makes the sequence labeling **globally optimal**, unlike softmax which makes independent token-level decisions.

---

### **6. What is the BIO tagging scheme?**
BIO stands for:
- **B-** → Beginning of an entity (e.g., B-PER)  
- **I-** → Inside an entity (e.g., I-PER)  
- **O-** → Outside any named entity  
It helps represent multi-word entities like “New York City” (B-LOC, I-LOC, I-LOC).

---

### **7. What metrics are used to evaluate NER models?**
- **Precision** – Correctly predicted entities / Total predicted entities  
- **Recall** – Correctly predicted entities / Total actual entities  
- **F1-Score** – Harmonic mean of precision and recall  
- **Accuracy** – Percentage of correctly labeled tokens  

---

### **8. What are the challenges in Named Entity Recognition?**
- **Ambiguity** – Same word may have different meanings (e.g., “Apple” can be a fruit or company).  
- **Nested entities** – One entity within another (e.g., “Bank of America Tower”).  
- **Domain shift** – Model trained on one domain may perform poorly on another.  
- **Out-of-vocabulary words** – Entities not seen during training.  

---

### **9. How does padding help in LSTM training?**
LSTM models require fixed-length input sequences.  
**Padding** adds dummy tokens (e.g., `__PAD__`) to shorter sequences so that all inputs in a batch have equal length, allowing efficient matrix operations during training.

---

### **10. What causes the vanishing gradient problem and how does LSTM solve it?**
In RNNs, gradients can shrink exponentially during backpropagation through time, preventing effective learning of long-range dependencies.  
LSTM solves this using **cell states and gating mechanisms**, which control the flow of information and preserve important features over long sequences.

---

### **11. Why combine LSTM with CRF for NER?**
- **LSTM** captures *contextual dependencies* within the sentence.  
- **CRF** ensures *label consistency* across the sequence.  
Together, they produce more accurate and linguistically valid entity predictions.

---

### **12. What were the final results of your model?**
- **Test Accuracy:** 98.42%  
- **Loss:** 0.0535  
This indicates that the model effectively captured both contextual and structural patterns for precise entity recognition.

---

### **13. Mention a few real-world applications of NER.**
- **Chatbots & Virtual Assistants** (extracting names, locations, dates)  
- **Search Engines** (query understanding)  
- **Resume Parsing** (identifying skills, education, experience)  
- **Information Retrieval** (summarizing named entities from documents)  

---

### **14. Explain the importance of contextual embeddings in NER.**
Contextual embeddings (like those from LSTMs or Transformers) capture the **meaning of a word based on its context**, enabling models to differentiate between polysemous words such as *“Apple (fruit)”* and *“Apple (company)”*.

---

### **15. What future improvements can be done on the model?**
- Use **pretrained embeddings** (e.g., GloVe, Word2Vec).  
- Upgrade to **Transformer-based architectures** (e.g., BERT, RoBERTa).  
- Apply **transfer learning** for domain-specific NER tasks.  
- Handle **nested and overlapping entities** using advanced models.

---

🧾 **End of Viva Q&A**
