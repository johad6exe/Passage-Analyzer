# AI Book Passage Analyzer

## Overview
This application is a highly modular, Streamlit-based Python tool designed to analyze literary passages. The script processes text to provide accurate word counts, predominant emotional analysis, intelligent book attributions, and concise summaries.

## Features & Architecture

To ensure speed, accuracy, and adherence to assignment constraints, the app utilizes the following architecture:

 **Edge-Case Rigorous Word Count:** Instead of naive whitespace splitting, the application utilizes the regex `\w+(?:['\-]\w+)*`. This successfully isolates alphanumeric words while properly handling internal contractions (e.g., "don't") and hyphenated structures (e.g., "well-known") as single words, while strictly ignoring trailing punctuation.
**Two-Tier Book Attribution Engine:** 
    **Tier 1 (Heuristics):** Applies regex word-boundary mapping to instantly identify high-confidence keyword/character hints (e.g., "Santiago", "Boo Radley") mapping to exact books. This saves compute time and tokens.
    **Tier 2 (LLM Generation):** Leverages the Groq API (`openai/gpt-oss-20b`) to complete the attribution list based on thematic and stylistic reasoning.
**Real-Time UI Streaming:** The Groq inference call concatenates emotion detection, book attribution, and summarization into a single prompt. The response is streamed token-by-token directly to the Streamlit UI via Python generators.

## 🛠️ Prerequisites

Make sure you have the following installed on your system:
- Virtual environment to avoid dependency errors.
- Python 3.10 or higher.
- pip (Python package installer)
- A valid **Groq API Key** in the .env file

## ⚙️ Installation & Setup

**1. Prepare the project directory**
Ensure the main script (`passage_analyzer.py`) is in your current working directory.

**2. Install required dependencies**
Run the following commands in your terminal to install dependencies and run the streamlit app: 

pip install -r requirements.txt
streamlit run passage_analyzer.py