# Policy Document Q&A System (RAG)

A retrieval-augmented generation system that answers questions over HR policy documents, with source citations (file name and page number).

## Tech Stack
Python · LangChain · Chroma · BGE embeddings (bge-small-zh-v1.5) · Qwen2.5-7B via Ollama · PyMuPDF

## Features
- Loads PDF/TXT documents and keeps file and page metadata for citations
- Preprocessing that restores clause boundaries in PDF text before chunking
- Answers constrained to retrieved context; out-of-scope questions are declined
- Retrieval evaluation (Hit@1 / Hit@3) on a 42-question evaluation set

## Results
| Chunk size | Chunks | Hit@1 | Hit@3 |
|-----------:|-------:|------:|------:|
| 50  | 244 | 73.8% | 85.7%  |
| 100 | 95  | 95.2% | 95.2%  |
| **200** | **45** | **95.2%** | **97.6%** |
| 400 | 26  | 85.7% | 95.2%  |
| 800 | 14  | 83.3% | 100.0% |

200-character chunks gave the best balance. Page-level chunks (800) reach 100% Hit@3 only by retrieving a large share of the corpus, and rank the correct chunk first less often.

Fixing PDF preprocessing (clauses were fragmented mid-sentence) improved Hit@1 from 88.1% to 95.2% at chunk size 200.

## Data
The knowledge base consists of four synthetic HR policy documents (14 pages) for a fictional company, written for this project. The evaluation set pairs each question with an answer keyword and its source file.

## Run
1. Install [Ollama](https://ollama.com) and run `ollama pull qwen2.5:7b`
2. `pip install -r requirements.txt`
3. Open `06_RAG的使用.ipynb` and run all cells

## Limitations
- Small evaluation set (42 questions), written by the same author as the documents
- The 7B model sometimes cites the wrong source number
