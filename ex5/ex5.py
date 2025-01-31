import wikipedia
import spacy
import random
from collections import defaultdict
import google.generativeai as genai
import re

# Load Spacy NLP model
nlp = spacy.load("en_core_web_sm")

# Set Gemini API key
genai.configure(api_key=key)


def get_wikipedia_content(title):
    """Fetch Wikipedia page content."""
    return wikipedia.page(title, auto_suggest=False).content


# POS-Tag-Based Extractor
def extract_pos_triplets(text):
    """Extract (Subject, Relation, Object) triplets based on POS tagging."""
    doc = nlp(text)
    triplets = []

    proper_nouns = [token for token in doc if token.pos_ == "PROPN"]

    for i in range(len(proper_nouns) - 1):
        subj, obj = proper_nouns[i], proper_nouns[i + 1]
        relation_tokens = [token for token in doc[subj.i + 1: obj.i] if token.pos_ in {"VERB", "ADP"}]

        if relation_tokens:
            relation = " ".join([token.text for token in relation_tokens])
            triplets.append((subj.text, relation, obj.text))

    return triplets


# Dependency-Tree-Based Extractor
def extract_dependency_triplets(text):
    """Extract (Subject, Relation, Object) triplets based on dependency parsing."""
    doc = nlp(text)
    triplets = []

    proper_heads = {token: set([token] + [child for child in token.children if child.dep_ == "compound"])
                    for token in doc if token.pos_ == "PROPN" and token.dep_ != "compound"}

    for h1, p1 in proper_heads.items():
        for h2, p2 in proper_heads.items():
            if h1 == h2:
                continue

            # Condition #1: Subject and Object share the same head
            if h1.head == h2.head and h1.dep_ == "nsubj" and h2.dep_ == "dobj":
                triplets.append(
                    (" ".join([t.text for t in p1]), h1.head.text, " ".join([t.text for t in p2])))

            # Condition #2: Prepositional object relationship
            if h1.head == h2.head.head and h1.dep_ == "nsubj" and h2.head.dep_ == "prep" and h2.dep_ == "pobj":
                triplets.append((" ".join([t.text for t in p1]), f"{h1.head.text} {h2.head.text}",
                                 " ".join([t.text for t in p2])))

    return triplets


# Evaluation Function
def evaluate_extractors(pages):
    """Run both extractors on given Wikipedia pages and evaluate output."""
    results = {}

    for page in pages:
        text = get_wikipedia_content(page)

        pos_triplets = extract_pos_triplets(text)
        dep_triplets = extract_dependency_triplets(text)

        results[page] = {
            "POS-Based": len(pos_triplets),
            "Dependency-Based": len(dep_triplets),
            "POS Sample": random.sample(pos_triplets, min(5, len(pos_triplets))),
            "Dependency Sample": random.sample(dep_triplets, min(5, len(dep_triplets)))
        }

    return results


# Wikipedia Pages to Evaluate
pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]

# Run Evaluation
evaluation_results = evaluate_extractors(pages)

# Print Results
for page, result in evaluation_results.items():
    print(f"Page: {page}")
    print(f"Total POS-Based Triplets: {result['POS-Based']}")
    print(f"Total Dependency-Based Triplets: {result['Dependency-Based']}")
    print(f"Sample POS-Based Triplets: {result['POS Sample']}")
    print(f"Sample Dependency-Based Triplets: {result['Dependency Sample']}")
    print("-" * 80)


def clean_response(response_text):
    """Cleans the LLM response by extracting only triplets and removing unnecessary formatting."""
    # Remove Markdown JSON block if present (```json ... ```)
    response_text = re.sub(r"```json\n(.*?)\n```", r"\1", response_text, flags=re.DOTALL).strip()

    # Extract potential triplets from structured text
    triplets = []
    for line in response_text.split("\n"):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):  # Ensuring it's a list-like format
            elements = line[1:-1].split(",")  # Split by commas, assuming simple text triplets
            elements = [e.strip().strip('"') for e in elements]  # Clean spaces and quotes
            if len(elements) == 3:
                triplets.append(tuple(elements))

    return triplets


def extract_llm_triplets(text):
    """Extract (Subject, Relation, Object) triplets using Google's Gemini API call."""
    prompt = f"""
    Extract (Subject, Relation, Object) triplets from the following text:
    {text}

    Respond only with structured triplets in this format:
    ["Subject1", "Relation1", "Object1"]
    ["Subject2", "Relation2", "Object2"]
    Do not add any explanations or extra text.
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    if not response or not response.text:
        print("Warning: Empty response from Gemini API.")
        return []

    return clean_response(response.text)


def evaluate_llm_extraction(pages):
    """Run the LLM extractor on Wikipedia pages and evaluate output."""
    results = {}

    for page in pages:
        text = get_wikipedia_content(page)
        llm_triplets = extract_llm_triplets(text)
        results[page] = {
            "Total LLM-Based Triplets": len(llm_triplets),
            "Sample LLM Triplets": llm_triplets[:5]  # Take first 5 as sample
        }

    return results


# Wikipedia Pages to Evaluate
pages = ["Donald Trump", "Ruth Bader Ginsburg", "J. K. Rowling"]

# Run LLM-Based Evaluation
evaluation_results = evaluate_llm_extraction(pages)

# Print Results
for page, result in evaluation_results.items():
    print(f"Page: {page}")
    print(f"Total LLM-Based Triplets: {result['Total LLM-Based Triplets']}")
    print(f"Sample LLM-Based Triplets: {result['Sample LLM Triplets']}")
    print("-" * 80)
