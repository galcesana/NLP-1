import re

# Define a function to map words to pseudo-words
def map_to_pseudo_word(word):
    if re.match(r'^[A-Z][a-z]+$', word):  # Proper nouns (capitalized)
        return "<PROPER_NOUN>"
    elif re.match(r'^[0-9]+$', word):  # Numbers
        return "<NUMBER>"
    elif re.match(r'^[A-Z]+$', word):  # Acronyms
        return "<ACRONYM>"
    elif re.match(r'.*ing$', word):  # Words ending in "ing"
        return "<VERB_ING>"
    elif re.match(r'.*ed$', word):  # Words ending in "ed"
        return "<VERB_ED>"
    elif re.match(r'.*ly$', word):  # Words ending in "ly"
        return "<ADVERB_LY>"
    elif re.match(r'.*ion$', word):  # Words ending in "ion"
        return "<NOUN_ION>"
    elif re.match(r'.*s$', word):  # Plurals
        return "<PLURAL>"
    else:
        return "<UNKNOWN>"
