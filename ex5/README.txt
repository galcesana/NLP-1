# Wikipedia Triplet Extractor

This project extracts (Subject, Relation, Object) triplets from Wikipedia articles using:
- **POS-Tag-Based Extraction** (uses spaCy for part-of-speech tagging).
- **Dependency-Based Extraction** (uses dependency parsing for structured extraction).
- **LLM-Based Extraction** (uses Gemini API for context-aware triplets).

## Usage
1. Install dependencies: `pip install wikipedia-api spacy google-generativeai`
2. Run `ex5.py` to extract and evaluate triplets.

**Note:** Requires a Google Gemini API key for LLM-based extraction.
