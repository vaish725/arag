**A-RAG Implementation Plan**

_Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces_

Based on: arXiv 2602.03442 · February 2026

Dataset: HotpotQA · Stack: Python · LangGraph · FAISS · BM25 · RAGAS · Streamlit

**What this document is**

A step-by-step engineering plan for implementing A-RAG - a state-of-the-art agentic RAG system published in February 2026 - as a portfolio project targeting software engineer / AI engineer roles. Covers architecture, full tech stack, phased implementation, dataset setup, evaluation, and how to present it in interviews.

**Total estimated build time: 5-7 days of focused work.**

# **1\. Background - Why A-RAG?**

Traditional RAG systems follow one of two fixed patterns: (1) single-shot retrieval - embed the query, pull the top-k chunks, feed them to the LLM, done - or (2) predefined workflow RAG where a human-designed pipeline decides the retrieval strategy in advance. Neither approach lets the model itself participate in retrieval decisions.

A-RAG (arXiv 2602.03442, February 2026) solves this by exposing three retrieval tools directly to the LLM and letting it decide, at each step, which tool to call, with what query, and when it has gathered enough evidence to answer. The model becomes an active agent in its own retrieval loop - not just a consumer of pre-retrieved context.

## **1.1 Key innovations over prior RAG**

| **Naive RAG**            | One embedding call → top-k chunks → answer. No iteration, no adaptation.                                                              |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Workflow RAG**         | Human-defined multi-step pipeline. Rigid - cannot adapt based on what it finds.                                                       |
| **A-RAG (this project)** | LLM chooses between keyword, semantic, and chunk-read tools. Iterates freely. Deduplicates via context tracker. Stops when confident. |

On multi-hop open-domain QA benchmarks, A-RAG outperforms naive RAG by ~13 percentage points while using comparable or fewer retrieved tokens - proving that smarter retrieval decisions beat brute-force context stuffing.

# **2\. Architecture Overview**

The system has three layers: the index layer (what gets built offline), the tool layer (the three retrieval interfaces), and the agent loop (the LLM reasoning and calling tools iteratively).

## **2.1 The three retrieval tools**

| **keyword_search(query, k)**  | BM25 lexical search over raw text. Fast. Best for specific names, IDs, dates, exact phrases. Returns chunk IDs + snippets.              |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **semantic_search(query, k)** | Embed query → FAISS cosine similarity. Best for conceptual, paraphrased, or thematic queries. Returns most semantically similar chunks. |
| **chunk_read(chunk_id)**      | Given a chunk ID found by either search, return its full surrounding context (adjacent chunks). Used to "zoom in" after finding a lead. |

## **2.2 Agent loop + context tracker**

The LLM runs in a ReAct-style loop: receive question → think → call a tool → observe result → think → call another tool or emit final answer. Two critical guardrails wrap this loop:

- Context tracker: maintains a set of seen chunk IDs. Any chunk already retrieved is excluded from future results, preventing the agent from re-reading the same content and wasting tokens.
- Budget guard: caps at 10 tool calls and ~4000 tokens of retrieved context total. If the budget is hit, the agent is forced to answer with what it has.

## **2.3 Data flow diagram**

User question

│

▼

┌─────────────────────────────────────────────────────────────┐

│ LLM Agent Loop (LangGraph ReAct) │

│ │

│ Think: which tool do I need? │

│ │ │

│ ├── keyword_search(q) ──► BM25 Index │

│ │ ▼ chunk IDs + snippets │

│ ├── semantic_search(q) ──► FAISS Vector Index │

│ │ ▼ similar chunks │

│ └── chunk_read(id) ──► Raw chunk store │

│ ▼ full context │

│ │

│ Context Tracker (dedup seen chunk IDs) │

│ Budget Guard (max 10 steps, 4k token cap) │

│ │

│ Repeat until: confident OR budget exhausted │

└─────────────────────────────────────────────────────────────┘

│

▼

Final answer (grounded in retrieved evidence)

# **3\. Full Tech Stack**

## **3.1 Core dependencies**

| **Python**              | 3.11+ (use pyenv or conda)                                                                               |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| **LangGraph**           | Agent loop orchestration - nodes, edges, conditional routing. Built by LangChain team, production-grade. |
| **LangChain-Anthropic** | Claude API bindings with tool-use support. Swap for langchain-openai if using OpenAI.                    |
| **FAISS**               | Facebook AI Similarity Search - in-process vector index. faiss-cpu for development.                      |
| **rank_bm25**           | Pure Python BM25 implementation. Fast enough for datasets up to ~100k docs.                              |
| **LlamaIndex**          | Document ingestion, chunking, and embedding pipeline. Handles PDF/text/JSON ingestion cleanly.           |
| **tiktoken**            | Token counting for budget guard - count tokens before passing context to LLM.                            |

## **3.2 Evaluation stack**

| **RAGAS**                  | pip install ragas - faithfulness, answer_relevancy, context_precision, context_recall metrics. LLM-as-judge framework, no ground-truth labels needed for most metrics. |
| -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **HotpotQA**               | The benchmark dataset used in the paper. Multi-hop QA requiring reasoning across multiple Wikipedia paragraphs. Download directly from HuggingFace datasets.           |
| **datasets (HuggingFace)** | pip install datasets - one-line download of HotpotQA train/validation splits.                                                                                          |
| **pandas + matplotlib**    | Store eval results as a DataFrame, plot comparison charts for the README.                                                                                              |

## **3.3 Observability and UI**

| **LangSmith**     | Free tier trace logging. Every tool call, token count, and latency captured automatically. Essential for demos and debugging.  |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **Streamlit**     | Chat UI with streaming. Displays tool calls in real time as the agent iterates. Run locally or deploy free on Streamlit Cloud. |
| **FastAPI**       | Optional - wrap the agent as a REST API (POST /query → JSON response with answer + trace).                                     |
| **python-dotenv** | Manage API keys across local dev and deployment.                                                                               |

## **3.4 Installation**

\# Create environment

python -m venv .venv && source .venv/bin/activate

\# Core

pip install langgraph langchain-anthropic langchain-openai faiss-cpu rank_bm25

pip install llama-index llama-index-embeddings-openai tiktoken

\# Eval

pip install ragas datasets pandas matplotlib

\# UI + observability

pip install streamlit fastapi uvicorn python-dotenv langsmith

\# Dataset

python -c "from datasets import load_dataset; load_dataset('hotpot_qa', 'fullwiki')"

## **3.5 Environment variables**

\# .env

ANTHROPIC_API_KEY=sk-ant-...

OPENAI_API_KEY=sk-... # for embeddings (text-embedding-3-small)

LANGSMITH*API_KEY=ls*\_... # free at smith.langchain.com

LANGSMITH_PROJECT=arag-project

LANGCHAIN_TRACING_V2=true

# **4\. Implementation Phases**

The project is structured into 5 phases. Each phase produces something runnable before moving on. Do not skip to Phase 3 without completing Phases 1 and 2 - the index is the foundation everything else depends on.

| **Phase 1** | **Build the Hierarchical Index (Day 1-2)** |
| ----------- | ------------------------------------------ |

### **Goal**

Ingest HotpotQA documents, chunk them at two granularities, build a FAISS vector index, and build a BM25 keyword index. By the end of this phase you can run a standalone search against both indexes from a Python script.

### **Dataset: HotpotQA**

HotpotQA is the dataset used in the A-RAG paper. It contains 113k Wikipedia-based question-answer pairs specifically designed to require multi-hop reasoning - the answer cannot be found in a single document, forcing the agent to chain evidence from multiple sources. This is exactly the failure mode of naive RAG that A-RAG is designed to fix.

- Split to use: fullwiki validation set (7,405 QA pairs - large enough to evaluate, small enough to run without GPU).
- Each QA pair comes with supporting Wikipedia paragraphs - use these as your document corpus (~300k paragraphs).
- For faster development iteration: start with distractor set (10k examples) instead, then scale.

### **Chunking strategy**

- Fine-grained chunks: ~200 tokens each. These are what go into the FAISS vector index and BM25 index. Smaller chunks = more precise retrieval.
- Coarse chunks: ~800 tokens. These are what chunk_read returns - when the agent finds a fine chunk of interest, it reads the surrounding coarse chunk for full context.
- Use LlamaIndex SentenceSplitter with chunk_size=200, chunk_overlap=30 for fine chunks.
- Store chunk metadata: source doc ID, paragraph index, char offset. You need this to implement chunk_read (find adjacent chunks).

### **Index construction code**

from datasets import load_dataset

from llama_index.core import Document

from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.openai import OpenAIEmbedding

import faiss, pickle

from rank_bm25 import BM25Okapi

\# Load HotpotQA supporting facts as corpus

ds = load_dataset('hotpot_qa', 'fullwiki', split='validation')

\# Collect all unique paragraphs

paragraphs = \[\]

for ex in ds:

for title, sents in zip(ex\['context'\]\['title'\], ex\['context'\]\['sentences'\]):

paragraphs.append({'title': title, 'text': ' '.join(sents)})

\# Deduplicate by title

seen, corpus = set(), \[\]

for p in paragraphs:

if p\['title'\] not in seen:

seen.add(p\['title'\]); corpus.append(p)

\# Fine-grained chunking

splitter = SentenceSplitter(chunk_size=200, chunk_overlap=30)

chunks = \[\] # list of {id, text, source_title, coarse_text}

for doc in corpus:

nodes = splitter.get_nodes_from_documents(\[Document(text=doc\['text'\])\])

for i, node in enumerate(nodes):

chunks.append({'id': f"{doc\['title'\]}\_{i}",

'text': node.text,

'source': doc\['title'\],

'coarse': doc\['text'\]}) # full para for chunk_read

\# Build BM25 index

tokenized = \[c\['text'\].lower().split() for c in chunks\]

bm25 = BM25Okapi(tokenized)

pickle.dump((bm25, chunks), open('bm25_index.pkl', 'wb'))

\# Build FAISS index

embed_model = OpenAIEmbedding(model='text-embedding-3-small')

embeddings = embed_model.get_text_embedding_batch(\[c\['text'\] for c in chunks\])

import numpy as np

matrix = np.array(embeddings).astype('float32')

index = faiss.IndexFlatIP(matrix.shape\[1\]) # inner product = cosine on normalized vecs

faiss.normalize_L2(matrix)

index.add(matrix)

faiss.write_index(index, 'faiss_index.bin')

pickle.dump(chunks, open('chunks.pkl', 'wb'))

print(f'Indexed {len(chunks)} chunks from {len(corpus)} documents')

### **Validation checkpoint**

- Run a test BM25 search for the query 'Scott Derrickson director' - verify you get relevant chunks about the director.
- Run a test semantic search for 'Who directed the Marvel film about a sorcerer?' - verify similar results.
- Check chunk count: expect ~500k-800k fine chunks from the full validation set corpus.

| **Phase 2** | **Implement the Three Retrieval Tools (Day 2-3)** |
| ----------- | ------------------------------------------------- |

### **Goal**

Wrap the BM25 and FAISS indexes as callable Python functions, then expose them as LangChain/LangGraph Tool objects with proper JSON schemas. By the end of this phase you can call each tool from a Python REPL and see results.

### **Tool 1: keyword_search**

import pickle

from langchain_core.tools import tool

bm25, chunks = pickle.load(open('bm25_index.pkl', 'rb'))

@tool

def keyword_search(query: str, k: int = 5) -> list\[dict\]:

'''BM25 lexical search. Use for specific names, dates, IDs, exact phrases.

Returns a list of {chunk_id, text, score} dicts.'

'''

tokens = query.lower().split()

scores = bm25.get_scores(tokens)

top_k = sorted(enumerate(scores), key=lambda x: -x\[1\])\[:k\]

return \[{'chunk_id': chunks\[i\]\['id'\],

'text': chunks\[i\]\['text'\]\[:300\],

'score': round(float(s), 3)} for i, s in top_k if s > 0\]

### **Tool 2: semantic_search**

import faiss, pickle, numpy as np

from llama_index.embeddings.openai import OpenAIEmbedding

faiss_index = faiss.read_index('faiss_index.bin')

chunks = pickle.load(open('chunks.pkl', 'rb'))

embed_model = OpenAIEmbedding(model='text-embedding-3-small')

@tool

def semantic_search(query: str, k: int = 5) -> list\[dict\]:

'''Vector similarity search. Use for conceptual or paraphrased queries.

Returns semantically similar chunks as {chunk_id, text, score} dicts.'

'''

vec = np.array(embed_model.get_text_embedding(query)).astype('float32').reshape(1, -1)

faiss.normalize_L2(vec)

scores, indices = faiss_index.search(vec, k)

return \[{'chunk_id': chunks\[idx\]\['id'\],

'text': chunks\[idx\]\['text'\]\[:300\],

'score': round(float(scores\[0\]\[i\]), 3)}

for i, idx in enumerate(indices\[0\]) if idx >= 0\]

### **Tool 3: chunk_read**

@tool

def chunk_read(chunk_id: str) -> dict:

'''Read the full context surrounding a chunk found by search.

Pass a chunk_id from keyword_search or semantic_search.

Returns the complete source paragraph for full context.'

'''

chunk = next((c for c in chunks if c\['id'\] == chunk_id), None)

if not chunk:

return {'error': f'chunk_id {chunk_id!r} not found'}

return {'chunk_id': chunk_id,

'source': chunk\['source'\],

'full_context': chunk\['coarse'\]} # full surrounding paragraph

### **Validation checkpoint**

- Call keyword_search('Scott Derrickson') directly in Python - confirm non-empty results.
- Call semantic_search('director of Doctor Strange') - confirm overlapping but not identical results to keyword search.
- Take a chunk_id from search results → call chunk_read(chunk_id) → confirm you get a longer context window.
- Verify all three return JSON-serializable dicts - LangGraph requires this.

| **Phase 3** | **Build the Agent Loop + Context Tracker (Day 3-4)** |
| ----------- | ---------------------------------------------------- |

### **Goal**

Wire the three tools into a LangGraph ReAct agent. Add the context tracker and budget guard. By the end of this phase you can pass any HotpotQA question to the agent and get a multi-step answer with full tool trace.

### **LangGraph state definition**

from langgraph.graph import StateGraph, END

from langchain_anthropic import ChatAnthropic

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from typing import TypedDict, Annotated

import operator

class AgentState(TypedDict):

messages: Annotated\[list, operator.add\] # full message history

seen_chunks: set # context tracker

step_count: int # budget guard

token_count: int # token budget guard

### **Agent node (LLM call)**

tools = \[keyword_search, semantic_search, chunk_read\]

llm = ChatAnthropic(model='claude-sonnet-4-5', temperature=0).bind_tools(tools)

SYSTEM_PROMPT = '''You are a research agent with three retrieval tools:

\- keyword_search: for specific names, dates, exact terms (use BM25)

\- semantic_search: for concepts and paraphrased queries (use vectors)

\- chunk_read: to get full context after finding a relevant chunk_id

Strategy: start broad (semantic or keyword), then zoom in (chunk_read).

For multi-hop questions: answer one hop at a time, use findings to guide next search.

Stop searching when you have sufficient evidence. Be concise in your final answer.'''

def call_agent(state: AgentState) -> AgentState:

response = llm.invoke(\[

{'role': 'system', 'content': SYSTEM_PROMPT},

\*state\['messages'\]

\])

return {'messages': \[response\], 'step_count': state\['step_count'\] + 1}

### **Tool execution node with context tracker**

import json, tiktoken

enc = tiktoken.encoding_for_model('gpt-4') # close enough for budgeting

def execute_tools(state: AgentState) -> AgentState:

last_msg = state\['messages'\]\[-1\]

tool_messages = \[\]

new_chunks = set(state\['seen_chunks'\])

total_tokens = state\['token_count'\]

for tool_call in last_msg.tool_calls:

fn = {'keyword_search': keyword_search,

'semantic_search': semantic_search,

'chunk_read': chunk_read}\[tool_call\['name'\]\]

result = fn.invoke(tool_call\['args'\])

\# Context tracker: filter out already-seen chunks

if isinstance(result, list):

result = \[r for r in result if r\['chunk_id'\] not in new_chunks\]

for r in result: new_chunks.add(r\['chunk_id'\])

elif isinstance(result, dict) and 'chunk_id' in result:

if result\['chunk_id'\] in new_chunks:

result = {'info': 'Already retrieved this chunk'}

else:

new_chunks.add(result\['chunk_id'\])

result_str = json.dumps(result)

total_tokens += len(enc.encode(result_str))

tool_messages.append(ToolMessage(content=result_str,

tool_call_id=tool_call\['id'\]))

return {'messages': tool_messages,

'seen_chunks': new_chunks,

'token_count': total_tokens}

### **Graph wiring + budget routing**

MAX_STEPS = 10

MAX_TOKENS = 4000

def should_continue(state: AgentState) -> str:

last = state\['messages'\]\[-1\]

\# Stop if: no tool calls, or budget exceeded

if not getattr(last, 'tool_calls', None):

return 'end'

if state\['step_count'\] >= MAX_STEPS or state\['token_count'\] >= MAX_TOKENS:

return 'end'

return 'tools'

graph = StateGraph(AgentState)

graph.add_node('agent', call_agent)

graph.add_node('tools', execute_tools)

graph.set_entry_point('agent')

graph.add_conditional_edges('agent', should_continue,

{'tools': 'tools', 'end': END})

graph.add_edge('tools', 'agent')

app = graph.compile()

def run_arag(question: str) -> str:

result = app.invoke({

'messages': \[HumanMessage(content=question)\],

'seen_chunks': set(),

'step_count': 0,

'token_count': 0,

})

return result\['messages'\]\[-1\].content

### **Validation checkpoint**

- Run run_arag('What award did the director of Titanic win at the Academy Awards?') - this is a two-hop question (find director → find award).
- Open LangSmith and verify you can see the full tool call trace for each run.
- Confirm the context tracker is working: the same chunk_id should never appear twice in a single run's tool outputs.
- Confirm budget guard fires if you set MAX_STEPS=2 and run a complex question.

| **Phase 4** | **Evaluation vs. Baseline (Day 4-5)** |
| ----------- | ------------------------------------- |

### **Goal**

Run three systems against the same 100 HotpotQA questions and measure with RAGAS. Produce a comparison table and chart that becomes the hero image in your README. This is the most important phase for interview credibility - you need numbers, not just a demo.

### **Three systems to compare**

| **System 1: Naive RAG**         | Single semantic_search call, top-5 chunks, direct LLM answer. No agent loop.                |
| ------------------------------- | ------------------------------------------------------------------------------------------- |
| **System 2: Naive Agentic RAG** | Same agent loop but only semantic_search tool available. Shows the value of tool diversity. |
| **System 3: A-RAG (yours)**     | Full implementation with all three tools + context tracker. This should win.                |

### **Evaluation script**

from ragas import evaluate

from ragas.metrics import faithfulness, answer_relevancy, context_precision

from datasets import Dataset

def evaluate_system(system_fn, questions, gold_answers, contexts_list):

"""system_fn(question) -> answer string"""

answers = \[system_fn(q) for q in questions\]

data = {

'question': questions,

'answer': answers,

'contexts': contexts_list, # list of lists of retrieved strings

'ground_truth': gold_answers,

}

return evaluate(Dataset.from_dict(data),

metrics=\[faithfulness, answer_relevancy, context_precision\])

\# Sample 100 questions from HotpotQA validation set

from datasets import load_dataset

ds = load_dataset('hotpot_qa', 'fullwiki', split='validation').select(range(100))

questions = \[ex\['question'\] for ex in ds\]

gold_answers = \[ex\['answer'\] for ex in ds\]

\# Run all three systems and collect results

results = {

'naive_rag': evaluate_system(naive_rag_fn, questions, gold_answers, naive_contexts),

'naive_agentic': evaluate_system(naive_agent_fn, questions, gold_answers, nagentic_ctx),

'arag': evaluate_system(run_arag, questions, gold_answers, arag_contexts),

}

\# Print comparison table

import pandas as pd

df = pd.DataFrame({k: {'Faithfulness': v\['faithfulness'\],

'Ans. Relevancy': v\['answer_relevancy'\],

'Context Precision': v\['context_precision'\]}

for k, v in results.items()}).T

print(df.to_string())

### **Also track: token efficiency**

For each system, log the total number of retrieved tokens per question. A-RAG should use fewer or comparable tokens while achieving higher accuracy - this is a key claim in the paper and a great interview talking point.

\# In your agent node, log token usage

import statistics

token_counts = {'naive_rag': \[\], 'arag': \[\]}

\# After each run:

token_counts\['arag'\].append(final_state\['token_count'\])

\# Report:

print(f"A-RAG avg tokens/query: {statistics.mean(token_counts\['arag'\]):.0f}")

print(f"Naive RAG avg tokens/query: {statistics.mean(token_counts\['naive_rag'\]):.0f}")

### **Validation checkpoint**

- A-RAG faithfulness should be >= Naive RAG faithfulness. If not, check that chunk_read is returning full enough context.
- A-RAG accuracy on HotpotQA exact match should be +8-15 points over Naive RAG (paper reports ~13pp).
- Export your results DataFrame as a PNG chart (matplotlib). This goes in your README.

| **Phase 5** | **Portfolio Polish + Deployment (Day 5-7)** |
| ----------- | ------------------------------------------- |

### **Streamlit UI**

\# app.py

import streamlit as st

from arag import run_arag

from naive_rag import run_naive_rag

st.title('A-RAG vs Naive RAG - HotpotQA Demo')

st.caption('Based on arXiv 2602.03442, February 2026')

question = st.text_input('Ask a multi-hop question:', placeholder=

'What is the nationality of the director of Titanic?')

if question:

col1, col2 = st.columns(2)

with col1:

st.subheader('Naive RAG')

with st.spinner('Retrieving...'):

st.write(run_naive_rag(question))

with col2:

st.subheader('A-RAG (this paper)')

with st.spinner('Agent reasoning...'):

st.write(run_arag(question))

st.subheader('Agent tool call trace')

st.info('Full trace available in LangSmith - link in README')

### **README structure (copy this)**

- H1: A-RAG Implementation - arXiv 2602.03442 (Feb 2026)
- 1-sentence hook: 'Agentic RAG where the LLM decides how to retrieve - outperforms naive RAG by 13pp on HotpotQA.'
- Architecture diagram (copy from Section 2.3 of this doc)
- Benchmark results table (your actual numbers from Phase 4)
- Live demo link (Streamlit Cloud or HuggingFace Spaces)
- Quick start: git clone → pip install → streamlit run app.py
- Paper citation block

### **Deployment**

\# Option A: Streamlit Cloud (free, easiest)

\# Push to GitHub → connect at share.streamlit.io → set secrets

\# Option B: HuggingFace Spaces (free)

\# Create a Space with SDK=streamlit, push code + requirements.txt

\# requirements.txt

langchain-anthropic>=0.3

langgraph>=0.2

faiss-cpu>=1.8

rank_bm25>=0.2

ragas>=0.1

streamlit>=1.35

datasets>=2.18

tiktoken>=0.7

python-dotenv>=1.0

# **5\. Interview Preparation**

For each question below, you should be able to answer from direct experience with your implementation - not from reading the paper. If you can't answer it, you haven't finished that phase yet.

## **5.1 Questions you will definitely get**

Q: Walk me through how A-RAG works differently from naive RAG. A: Naive RAG embeds the query once, retrieves top-k chunks, and feeds them to the LLM - the model has no say in what gets retrieved. A-RAG gives the model three tools and lets it reason about which one to call, iterate based on what it finds, and stop when it's confident. The key addition is the context tracker, which deduplicates seen chunks so the agent doesn't waste tokens re-reading the same content.

Q: Why three tools? Why not just semantic search? A: Each tool has a different failure mode. Semantic search misses exact-match queries - if you ask for 'Scott Derrickson' by name, an embedding search might return thematically similar directors instead of him specifically. BM25 nails exact matches but fails on paraphrased queries. chunk_read lets the agent drill into a specific document section once it has a lead. The agent learns when to use which - that's the core contribution.

Q: What's a context tracker and why does it matter? A: Without deduplication, the agent might call semantic_search('director of Titanic'), get chunk X, then call keyword_search('James Cameron director'), get chunk X again, and pass it to the LLM twice - burning tokens without adding information. The context tracker maintains a set of seen chunk IDs and filters them from all future tool results. This is why A-RAG achieves comparable or lower token usage than naive RAG despite iterating multiple times.

Q: How did you evaluate it? What metrics? A: I used RAGAS - faithfulness (does the answer follow from retrieved context?), answer relevancy (does the answer address the question?), and context precision (are the retrieved chunks actually relevant?). I also measured exact match accuracy against HotpotQA gold answers and tracked average tokens retrieved per query to verify efficiency claims from the paper.

Q: How does this relate to DeepSeek-R1 or test-time compute scaling? A: Both are about doing more computation at inference time to improve quality - just in different modalities. DeepSeek-R1 scales reasoning by generating longer chain-of-thought traces. A-RAG scales retrieval by letting the model make multiple adaptive retrieval calls. Both show that inference-time scaling is a more efficient path than simply training larger models. The connection interviewers want is: you understand the 2026 trend, not just the individual techniques.

## **5.2 Things to have ready to show**

| **Live demo**        | A running Streamlit app - show a multi-hop question where A-RAG gets it right and naive RAG gets it wrong.                            |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **LangSmith trace**  | Pull up a trace showing the agent's exact tool call sequence. Explain what each step is doing.                                        |
| **Benchmark chart**  | Your matplotlib comparison chart - faithfulness and accuracy across all 3 systems.                                                    |
| **GitHub README**    | Should be readable in 60 seconds by a mobile viewer. Architecture diagram + numbers + live link.                                      |
| **Token efficiency** | Know your average token count per query for A-RAG vs naive RAG. If you achieved lower tokens at higher accuracy, that's the headline. |

# **6\. Common Mistakes to Avoid**

| **Skipping Phase 4**              | The project is a demo without benchmarks. Interviewers can tell when numbers are missing. 100 questions takes ~30 min to evaluate - do it.                                                                                                            |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Using only one index**          | If you only have FAISS, you've built naive agentic RAG, not A-RAG. The BM25 index is required. The ablation study in the paper proves each tool contributes.                                                                                          |
| **No context tracker**            | Without deduplication, you'll find the agent retrieves the same chunk multiple times on complex queries. This bloats token usage and contradicts the paper's efficiency claim.                                                                        |
| **Ignoring failure modes**        | The paper has a failure mode appendix. Know the two main A-RAG failure modes: retrieval drift (agent chases a wrong lead) and early stopping (agent answers with insufficient evidence). Being able to cite these shows you read the paper seriously. |
| **Deploying without rate limits** | If you put a live demo online, add st.cache_resource on your index objects and limit to 10 questions/session - otherwise you'll burn your API quota in a demo.                                                                                        |
| **README without numbers**        | The most common portfolio mistake. 'I implemented A-RAG from the paper' with no benchmark table is an incomplete claim. Add your actual numbers even if they're slightly below paper numbers - honest results are fine.                               |

# **7\. Optional Extensions (Bonus Points)**

Do these only if Phases 1-5 are fully complete. Each extension adds 1-2 days but meaningfully differentiates you.

| **GraphRAG hybrid**           | Add a knowledge graph index (using the Microsoft GraphRAG library) as a fourth tool. Compare 4-tool A-RAG vs 3-tool on questions requiring thematic synthesis. |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Test-time compute scaling** | Replicate Section 5.1 of the paper: run A-RAG with max_steps = 3, 6, 10, 15 and plot accuracy vs steps. Shows you understand inference-time scaling.           |
| **FastAPI deployment**        | Wrap the agent as a REST API - POST /query returns JSON with answer, tool_trace\[\], token_count, latency_ms. Lets you show production thinking.               |
| **Streaming UI**              | Use LangGraph's streaming API to surface each tool call to the Streamlit UI in real time as it happens. Dramatically more impressive in demos.                 |
| **RAGAS CI**                  | Add a GitHub Actions workflow that runs the 100-question RAGAS eval on every push and posts results as a PR comment. Shows MLOps thinking.                     |

# **8\. Key Resources**

| **A-RAG paper**         | <https://arxiv.org/abs/2602.03442>                    |
| ----------------------- | ----------------------------------------------------- |
| **A-RAG official code** | <https://github.com/Ayanami0730/arag>                 |
| **HotpotQA dataset**    | <https://hotpotqa.github.io> / HuggingFace: hotpot_qa |
| **LangGraph docs**      | <https://langchain-ai.github.io/langgraph/>           |
| **RAGAS docs**          | <https://docs.ragas.io>                               |
| **LangSmith**           | <https://smith.langchain.com> (free tier)             |
| **FAISS**               | <https://github.com/facebookresearch/faiss>           |
| **rank_bm25**           | <https://github.com/dorianbrown/rank_bm25>            |
| **Streamlit Cloud**     | <https://share.streamlit.io> (free deployment)        |
| **HuggingFace Spaces**  | <https://huggingface.co/spaces> (free deployment)     |

**The goal**

When an interviewer asks 'tell me about a recent AI project,' you say: 'I implemented A-RAG - a paper from February 2026 that beats naive RAG by letting the model control its own retrieval strategy. I built it on HotpotQA with three tools, a context tracker, and a RAGAS eval harness. Here's the live demo and here are the numbers.' That answer takes 20 seconds. It is specific, grounded in a real paper, and demonstrably recent. That is the difference between getting the callback and not.