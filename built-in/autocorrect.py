"""
Autocorrect Module - Dictionary-based word correction
Implements Levenshtein distance for smart suggestions
"""

import difflib
from pathlib import Path

class AutoCorrect:
    def __init__(self, dictionary_path="built-in/dictionary.txt", enabled=True):
        self.enabled = enabled
        self.dictionary = set()
        
        if not self.enabled:
            return
            
        dict_file = Path(dictionary_path)
        if dict_file.exists():
            with open(dict_file, 'r', encoding='utf-8') as f:
                self.dictionary = {line.strip().upper() for line in f if line.strip()}
            print(f"[INFO] Autocorrect loaded {len(self.dictionary)} words")
        else:
            print(f"[WARN] Dictionary not found: {dictionary_path}")
            self.enabled = False
    
    def check_word(self, word):
        """Check if word exists in dictionary"""
        if not self.enabled or not word:
            return True
        return word.upper() in self.dictionary
    
    def get_suggestions(self, word, max_suggestions=3):
        """Get correction suggestions for misspelled word"""
        if not self.enabled or not word:
            return []
        
        word_upper = word.upper()
        if word_upper in self.dictionary:
            return []  # Word is correct
        
        # Use difflib for fuzzy matching
        matches = difflib.get_close_matches(
            word_upper, 
            self.dictionary, 
            n=max_suggestions, 
            cutoff=0.6
        )
        
        return matches
    
    def auto_correct(self, word):
        """Auto-correct word to best match"""
        if not self.enabled:
            return word
            
        suggestions = self.get_suggestions(word, max_suggestions=1)
        return suggestions[0] if suggestions else word
