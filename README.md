# Local RAG Toy Project 

This is a minimal Retrieval-Augmented Generation (RAG) pipeline using:
-  Ollama (`mxbai-embed-large`) for local embedding
-  FAISS as the vector store
-  Ollama (`dolphin-mistral`) for question answering
-  Movie reviews dataset as a toy corpus

## Features

- Fully local, GPU-optional RAG setup
- Ingests CSV with movie titles, reviews, and metadata
- Embeds documents and stores them in FAISS
- Retrieves relevant snippets to answer freeform questions
- Uses LangChain for chaining prompts with retrievers + models

---

## Setup

1. Install [Ollama](https://ollama.com) and pull the required models:

```sh
ollama pull mxbai-embed-large
ollama pull dolphin-mistral
```

2. Create a virtual environment (optional but recommended)  
```sh
python3 -m venv venv
source venv/bin/activate 
```

3. Install dependencies
```sh
pip install -r requirements.txt
```

4. Launch the RAG CLI:
```sh
python main.py
```

---

### 5. Optional Improvements (for a “Next Version” Branch)

| Feature | Description |
|--------|-------------|
| Web UI | Use Streamlit or Gradio for a friendly UI |
| Highlight source docs | Return metadata of matching docs |
| Add unit tests | Simple Pytest file that confirms embedding + retrieval |
| Add save/load via argparse | Support CLI flags like `--rebuild` |
| Filter by metadata | Enable retrieval scoped by `rating` or `date` |

Dataset
The dataset is located in:

bash
Copy
Edit
realistic_movie_reviews.csv

You can swap this file with any domain-specific .csv to experiment with custom data.

## Dataset
```sh
realistic_movie_reviews.csv
```
Each row includes:

- `Title`
- `Release Date`
- `Rating`
- `Review`

You can swap this file with any domain-specific `.csv` to experiment with other types of content (e.g., research papers, customer reviews, support logs).
---

## Usage Tips

Want to try sample questions?

Check out [`questions.md`](./questions.md) for ideas on what you can ask this RAG system — from basic rating lookups to opinion-based synthesis.

You can customize the dataset or prompt template to work on different domains like legal, academic, or technical review datasets.

--- 

## Contributions

Pull requests and improvements are welcome — this project is intended as a learning playground and a lightweight base for more advanced applications.

---

## Credits

Built with:

- [LangChain](https://github.com/langchain-ai/langchain)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com)

Inspired by @timdettmers's minimal RAG tutorial and other local AI projects.