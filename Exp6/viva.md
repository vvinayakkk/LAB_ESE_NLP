# # Viva Questions - LLM-Based NLP Applications
---

### 1. What is the main objective of your experiment?

**Answer:**  
The main objective is to demonstrate practical NLP applications using modern Large Language Models (LLMs), comparing open-source and proprietary offerings across multiple tasks: summarization, text classification, sentiment analysis, and machine translation. The experiment aims to show differences in performance, latency, usability, and output style between models you can run locally (Hugging Face Transformers) and large cloud-hosted models accessed via APIs (Google Gemini). It also illustrates engineering trade-offs (cost, compute, latency), prompting and pipeline utilities, and when to prefer one approach over another for production or research use-cases.

---

### 2. What are the two types of models you used? Give examples.

**Answer:**  
- **Open-Source models** (local/hosted by user): Examples used include `google-t5/t5-small` (summarization / text-to-text) and `facebook/bart-large-mnli` (zero-shot classification). These models are free to use; you download weights and run inference locally or on your own cloud/GPU instances. They require local compute (CPU/GPU), and their performance and latency depend on available hardware.
- **Proprietary models** (API/cloud): Example used is `gemini-2.5-flash` (Google Gemini family). These are accessed via cloud APIs, usually pay-as-you-go. They are often optimized for latency and scaling, may provide multimodal capabilities, and can be updated centrally by the provider.

---

### 3. Why do you measure latency in each task?

**Answer:**  
Latency measures inference speed and responsiveness, which is critical for user-facing systems (chatbots, real-time services) and for cost estimation (API usage time). Comparing latency between open-source and proprietary models helps evaluate:
- Whether on-device or hosted inference meets application requirements,
- The effect of model size on response times,
- Network overhead for API calls versus local compute overhead.  
Proprietary APIs often have optimized inference infrastructure that yields lower latency, while open-source models' latency depends on hardware (CPU vs GPU) and can be higher on constrained environments like Colab free tier.

---

### 4. Explain the difference between pipeline() and direct model usage.

**Answer:**  
- `pipeline(task)`: A high-level convenience API in Hugging Face that automatically loads a suitable model and tokenizer, handles preprocessing and postprocessing, and returns human-friendly outputs. It is ideal for rapid prototyping and simple experiments. For example, `pipeline("summarization")` handles text chunking, decoding, and returns summary strings.
- Direct model usage: Involves manually loading `AutoModel` and `AutoTokenizer`, tokenizing inputs, running `model(**inputs)`, and handling logits/decoding. This approach gives finer control over batching, custom decoding strategies, loss calculation, and integration with custom training loops. Direct usage is preferable when you need optimized performance, custom tokenization, or non-standard input/output processing.

---

### 5. Why do you prepend "summarize: " in T5 input?

**Answer:**  
T5 is a text-to-text framework: all tasks are expressed as natural language instructions prepended to the input. Prepending `"summarize: "` tells the model to perform summarization rather than another text-to-text transformation. This instruction conditions the model to generate output in the expected format and often improves generation fidelity because the model was trained on similar task-prefixed data.

---

### 6. What is Zero-Shot Classification? How does BART-MNLI do it?

**Answer:**  
Zero-shot classification assigns labels to text without labeled training examples for those labels. BART-large-MNLI (and similar models) are used via Natural Language Inference (NLI): treat the text as the premise and craft hypotheses like "This text is about technology." The model computes entailment/contradiction scores for each hypothesis. The label whose hypothesis receives the highest entailment score is treated as the best label. This method leverages the model's understanding of semantic relationships learned during NLI pretraining/fine-tuning.

---

### 7. Why does Gemini give top 3 labels while HF gives scores?

**Answer:**  
Hugging Face zero-shot pipelines return probabilistic scores because the underlying NLI model produces logits that can be converted to probabilities for each candidate label. Gemini, being a generative LLM, typically responds in natural language to prompts; unless explicitly requested for probabilities, it will output a ranked list or a textual explanation. To get top-K behavior from Gemini you instruct it to "Return top 3 categories" in the prompt, and it generates them. Generative models are not inherently calibrated probabilistic classifiers unless the API exposes probability estimates or special endpoints.

---

### 8. Which sentiment model did you use in Hugging Face?

**Answer:**  
The `pipeline("sentiment-analysis")` defaults to a model like `distilbert-base-uncased-finetuned-sst-2-english` (or similar), which is fine-tuned on the SST-2 dataset. This model provides binary or ternary sentiment labels (`POSITIVE` / `NEGATIVE`, sometimes `NEUTRAL` depending on the variant) with confidence scores. It is lightweight and optimized for general sentiment classification tasks.

---

### 9. Why does Gemini give an explanation along with sentiment?

**Answer:**  
Gemini is an instruction-following generative model; when prompted to classify and "briefly explain", it produces the label and a textual explanation of its reasoning because generation naturally supports free-form outputs. This makes the output more interpretable and useful for debugging or user-facing applications where an explanation adds transparency. The explanation is generated based on the LLM's internal knowledge and prompt context rather than explicit calibration or a separate explanation module.

---

### 10. Name the translation model used in Hugging Face.

**Answer:**  
The translation pipeline used is often backed by OPUS-MT models, e.g., `Helsinki-NLP/opus-mt-en-hi` for English → Hindi translation. These models are part of the MarianMT family and are trained on parallel corpora to perform machine translation between specific language pairs. They are efficient and designed for many language combinations.

---

### 11. Can Gemini do translation without fine-tuning?

**Answer:**  
Yes. Gemini is trained on large multilingual corpora and demonstrates few-shot and zero-shot multilingual capabilities. It can translate between many languages out-of-the-box through instruction prompts, without task-specific fine-tuning. The translation quality depends on training data coverage for the language pair; for major languages the quality is high, whereas for low-resource pairs it may be variable.

---

### 12. What are the advantages of using Gemini API?

**Answer:**  
- Speed: Cloud-optimized inference often yields lower latency than running large models locally.  
- Scalability: Managed backend scales automatically for high-load applications.  
- Multimodal capabilities: Gemini supports text + image and may provide richer multimodal understanding.  
- Up-to-date models: Proprietary services can roll out improvements centrally without user maintenance.  
- Ease of use: Simple endpoints like `generate_content()` reduce integration complexity.  
- Instruction following: Strong instruction-tuning often yields more helpful and controlled outputs.

---

### 13. What are the limitations of open-source models in your setup?

**Answer:**  
- Hardware constraints: On CPU or limited GPU instances (Colab free tier), large models are slow.  
- Model size vs quality trade-off: Smaller models (`t5-small`) are fast but less accurate; larger models need more memory.  
- Maintenance: Users must manage dependencies and updates.  
- Multimodality: Many HF pipelines focus on single modalities unless specific multimodal models are used.  
- Freshness: Models are static after download; they lack instantaneous access to updated knowledge unless re-trained or fine-tuned.

---

### 14. How would you extend this to Text Embedding & Clustering?

**Answer:**  
Use sentence embedding models such as `sentence-transformers/all-MiniLM-L6-v2` to convert textual inputs into dense vectors. Steps:
1. Load: `from sentence_transformers import SentenceTransformer` and `model = SentenceTransformer('all-MiniLM-L6-v2')`.
2. Encode: `embeddings = model.encode(sentences, convert_to_numpy=True)`.
3. Cluster: use `sklearn.cluster.KMeans` or `HDBSCAN` on embeddings to find groups, or compute nearest neighbors for semantic search.  
Embeddings enable similarity search, semantic clustering, and downstream unsupervised tasks.

---

### 15. How to build a Chatbot using Gemini?

**Answer:**  
Gemini provides chat-style APIs that maintain conversation history and context. Typical flow:
1. Initialize a chat session: `chat = model.start_chat()` (or the API-equivalent).
2. Send user messages: `response = chat.send_message("Hello!")`.
3. Store and manage context: the model keeps track of conversation turns or you provide a conversation buffer in the prompt.
4. Add system instructions for persona, constraints, and safety.
This yields a stateful, instruction-following chatbot that can be deployed with session and rate management.

---

### 16. How to do Image Understanding with Gemini?

**Answer:**  
Gemini supports multimodal inputs (text + images). Example approach:
1. Load an image (e.g., PIL): `img = Image.open("photo.jpg")`.
2. Send image + text: `response = model.generate_content(["Describe this image:", img])`.
3. Parse the model's output which may include captions, object lists, or answers to visual questions.  
This allows VQA, captioning, and multimodal reasoning with a single model endpoint.

---

### 17. Can Hugging Face do Image Understanding?

**Answer:**  
Yes. Hugging Face hosts many specialized vision models and pipelines:
- `pipeline("image-classification", model="google/vit-base-patch16-224")` for classification.
- `pipeline("image-segmentation")`, `pipeline("object-detection")`, or `pipeline("visual-question-answering", model="Salesforce/blip-vqa-base")` for VQA.  
However, these are specific models with narrower scopes; multimodal integration across modalities might require stitching different models or using multimodal HF models like CLIP or BLIP.

---

### 18. How to generate images using LLMs?

**Answer:**  
LLMs like Gemini generally output text; image generation is handled by specialized generative models. Options:
- Stable Diffusion: `from diffusers import StableDiffusionPipeline` and generate images from prompts.
- DALL·E or Imagen APIs (proprietary).  
Example with diffusers:
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe("A cat astronaut").images[0]
```
These models are optimized for image synthesis and accept natural-language prompts.

---

### 19. How to do Audio/Video Understanding?

**Answer:**  
Common tools and tasks:
- Speech-to-Text: `openai/whisper` (HF) or other ASR models for transcription.
- Audio classification: models like `MIT/ast-finetuned-audioset`.
- Video captioning & understanding: combine frame-level visual models and temporal models (e.g., video transformers or BLIP-style pipelines adapted to video).  
Gemini's support for audio/video varies by release; check API docs for multimodal audio/video capabilities. Example pipeline:
1. Extract audio frames → run ASR to transcribe.
2. Use transcripts + visual features to perform captioning or VQA on video frames.
Tools: `whisper` (ASR), `transformers`/`diffusers` for frame-level models, and libraries such as `ffmpeg` for preprocessing.

---

### 20. Summary: Open-Source vs Proprietary LLMs

**Answer:**  
A concise comparison table highlighting trade-offs:

| Feature | Hugging Face (Open-Source) | Google Gemini (Proprietary) |
|--------:|---------------------------:|-----------------------------:|
| Cost | Free models (compute cost only) | API costs (per request/usage) |
| Speed | Slower on limited hardware; depends on local compute | Fast due to cloud-optimized infra |
| Customization | High (fine-tune, modify) | Limited model-level customization |
| Multimodal | Limited unless specific models are used | Strong multimodal capabilities |
| Ease of Use | `pipeline()` easy; needs infra management | Simple API; managed service |
| Up-to-date | Static until re-trained | Centrally updated by provider |
| Privacy | Full control over data (on-prem) | Data sent to provider (check policies) |

---

### 21. How do you avoid hallucinations in LLM outputs?
**Answer:**  
- Use prompting strategies that constrain outputs (instructions, system messages).  
- Post-hoc verification with retrieval-augmented generation (RAG) or grounding outputs on trusted sources.  
- Use smaller, more deterministic decoding (beam search, nucleus sampling with low temperature) and add validation checks.
