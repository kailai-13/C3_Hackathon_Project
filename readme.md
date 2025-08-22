# README.md

## Concordia Buyer Agent Project

This repository contains an **AI-powered buyer agent** built with Google DeepMind's Concordia framework. The agent negotiates for perishable goods (such as mangoes) against an intelligent seller, aiming to maximize savings, maintain personality consistency, and close deals efficiently.

***

## 📦 Submission Files

- **buyer_agent.py**  
  Core Python implementation of the Concordia-based buyer agent, including negotiation policy, memory, personality component, and testing routines.

- **personality_config.json**  
  JSON definition of your agent’s core traits, negotiation style, and communication catchphrases.

- **strategy.md**  
  One-page markdown document explaining your agent’s negotiation strategy, rationale for personality, decision logic, and insights from testing.

- **requirements.txt**  
  List of Python package dependencies required to run the agent.

***

## 🚀 Quickstart

### 1. Clone Repository
```bash
git clone <repo-url>
cd <repo-directory>
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```
Make sure you have the Concordia framework and sentence-transformers installed.

### 3. Run Tests
```bash
python buyer_agent.py                # Runs built-in test scenarios with conversation output
python buyer_agent.py --silent       # Runs tests with minimal output
python buyer_agent.py --single       # Run an interactive single negotiation
python buyer_agent.py --debug        # Run performance benchmarks
python buyer_agent.py --model llama3 # Choose LLM model (default: llama3:8b)
```

***

## 🧠 Project Overview

- **Concordia Framework:**  
  Used for composable agent components, memory, personality management, and context-aware negotiation.

- **Agent Personality:**  
  Defined in personality_config.json and embedded in all negotiation messages for consistency.

- **Negotiation Strategy:**  
  - Anchors opening offers strategically below market price
  - Uses adaptive concession logic as rounds progress
  - Accepts fair deals quickly
  - Tracks whole negotiation via memory
  - Never exceeds buyer budget

- **LLM Integration:**  
  Generates dialogue in context, maintaining character and adapting to seller moves.

***

## 📝 Files Description

| File                    | Description                                              |
|-------------------------|----------------------------------------------------------|
| buyer_agent.py          | Main agent implementation and test routines              |
| personality_config.json | Agent’s core traits, negotiation style, catchphrases     |
| strategy.md             | Explanation of agent rationale, strategy, and insights   |
| requirements.txt        | Python library dependencies                              |

***

## ⚙️ Code Standards

- Modular, well-documented code
- Type hints used throughout
- All key methods contain docstrings
- No hard-coded negotiation values

***

## 📋 How to Submit

See your challenge instructions for the submission command, typically:
```bash
python submit_project.py --name "Your Name" --email "your.email@example.com"
```

***

## ❓ FAQ

- Q: Where do I customize agent personality?
  - A: In `personality_config.json` (traits, style, catchphrases).
- Q: How do I tune negotiation policy?
  - A: Policy config is set in `buyer_agent.py` (`_PolicyConfig` section).
- Q: Where are test scenarios defined?
  - A: In the testing section of `buyer_agent.py`.

***

**Good luck! May your agent negotiate with skill and style.**

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/70191914/c01d4a6f-b75b-4171-8a92-94f469379795/paste.txt)
[2](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/images/70191914/854e5aae-96cb-417e-99b7-e1f8d60ccc89/WhatsApp-Image-2025-08-22-at-15.01.44_62641df9.jpg)