import unittest
import re

input = """
                 Software Developer

              Strong Python "skills" and     !solid understanding..... of data structures,,, and algorithms.!
"""

expected = "Software Developer Strong Python skills and solid understanding of data structures and algorithms"

def clean_text(text: str) -> str:
    # Remove punctuation and non-letters, keep only letters + spaces
    cleaned = re.sub(r"[^A-Za-z\s]", "", text)

    # Normalize multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned

class TestCleanTest(unittest.TestCase):

    def test_clean_text(self):
        result = clean_text(input)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()

# run
# python -m unittest test_clean_text.py
