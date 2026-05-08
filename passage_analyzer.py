import streamlit as st
import re
import os
from groq import Groq
from typing import Optional, Generator
from dotenv import load_dotenv

load_dotenv()  # Loading environment variables from .env file

# Using an inference model which works well for reasoning, , emotion analysis and summarization yet lightweight for fast inference.
DEFAULT_MODEL = "openai/gpt-oss-20b" 

# --- MODULE 1: TEXT PROCESSING & HEURISTICS ---

def count_words(text: str) -> int:
    """Accurately count words using regex, handles punctuations and special characters intelligently."""
    # Matches sequences of word characters, ignoring punctuation attached to words.
    words = re.findall(r"\w+(?:['\-]\w+)*", text)
    return len(words)

def tier_1_book_heuristic(text: str) -> Optional[str]:
    """
    Tier 1 Approach: Uses keyword heuristics from the books to find exact/highly likely matches.
    """
    text_lower = text.lower()
    
    # Heuristic dictionary matching book titles to their strong contextual keywords/character names
    heuristics = {
        "The Alchemist by Paulo Coelho": ["santiago", "alchemy", "personal legend", "melchizedek", "andalusian", "umm al-kiyaab"],
        "Man's Search for Meaning by Viktor Frankl": ["logotherapy", "concentration camp", "auschwitz", "frankl", "meaning in suffering", "capo"],
        "To Kill a Mockingbird by Harper Lee": ["scout", "atticus", "jem", "boo radley", "maycomb", "tom robinson", "mockingbird"]
    }
    
    for book, keywords in heuristics.items():
        # Using word boundaries (\b) to prevent partial word matches (e.g., 'scout' inside 'boy-scouts')
        if any(re.search(rf'\b{re.escape(kw)}\b', text_lower) for kw in keywords):
            return book
            
    return None

# --- MODULE 2: LLM INFERENCE ---

def stream_llm_analysis(passage: str, heuristic_match: Optional[str], api_key: str) -> Generator[str, None, None]:
    """
    Tier 2 Approach: Calls the Groq API to stream the emotion analysis, remaining books, and summary.
    """
    client = Groq(api_key=api_key)
    
    # Dynamically adjust the prompt based on whether Tier 1 found a match
    if heuristic_match:
        books_instruction = f"1. **{heuristic_match}** (Identified via contextual heuristics).\nSuggest 2 MORE possible books based on writing style, theme, and vocabulary."
    else:
        books_instruction = "Suggest 3 possible books this passage might belong to based on writing style, theme, and vocabulary."

    prompt = f"""You are an expert literary analyzer. Analyze the following literary passage and provide your response strictly using the markdown headings below.

### Predominant Emotion
Identify a primary emotion (e.g., joy, sadness, anger, fear, trust) conveyed in the passage, explain it briefly in 1 sentence.

### Possible Books
{books_instruction}
Provide a brief 1-sentence justification for the new suggestions.

### Summary
Provide a concise summary of the passage in EXACTLY 2-3 sentences.

Passage to analyze:
"{passage}"
"""

    # Stream the response from the Groq API
    try:
        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, # Low temperature for more analytical/factual output
            max_tokens=1024,
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"\n\n**API Error:** {str(e)}\n*Note: Ensure 'gpt-oss-20b' is a valid model ID on your Groq tier, or replace it with 'llama-3.1-8b-instant'.*"

# --- MODULE 3: STREAMLIT UI ---

def main():
    st.set_page_config(page_title="AI Literary Analyzer", page_icon="📚", layout="centered")
    
    st.title("📚 Book Passage Analyzer")
    st.markdown("Upload a text file or paste a passage below to extract word counts, emotions, book attributions, and a summary.")

    api_key = os.getenv("GroqAPI")  # Fetching the Groq API key from environment variables

    if api_key:
        st.toast("Groq API key loaded successfully.", icon="✅")
    else:
        st.toast("Groq API key not found. Please add it to the .env file.", icon="⚠️")

    # Input methods: File upload or text area
    input_method = st.radio("Choose input method:", ("Text Input", "File Upload (TXT)"), horizontal=True)
    
    passage = ""
    if input_method == "Text Input":
        passage = st.text_area("Paste your passage here:", height=200)
    else:
        uploaded_file = st.file_uploader("Choose a .txt file", type="txt")
        if uploaded_file is not None:
            # Decode bytes to string
            passage = uploaded_file.getvalue().decode("utf-8")
            st.info("File uploaded successfully. Scroll down to see the content and analyze.")
            with st.expander("View Uploaded Content"):
                st.write(passage)

    # Execution trigger
    if st.button("Analyze Passage", type="primary"):
        # Edge case: No API key
        if not api_key:
            st.error("Please make sure your API key is available in the .env file.")
            return
            
        # Edge case: Empty input
        if not passage.strip():
            st.warning("Please provide a passage to analyze.")
            return

        with st.spinner("Analyzing..."):
            # 1. Word Count (Standard Python/Regex)
            word_count = count_words(passage)
            st.success(f"**Word Count:** {word_count} words")
            
            # 2. Tier 1 Book Attribution (Heuristic)
            heuristic_match = tier_1_book_heuristic(passage)
            
            st.divider()
            
            # 3. LLM Inference (Streamed Emotion, Tier 2 Books, and Summary)
            st.subheader("🤖 AI Analysis")
            
            # Use Streamlit's built-in generator writing for real-time streaming
            response_generator = stream_llm_analysis(passage, heuristic_match, api_key)
            st.write_stream(response_generator)

if __name__ == "__main__":
    main()