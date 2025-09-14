import unittest
from utils.utils import clean_text

input = """
                 Software Developer

              Strong Python "skills" and     !solid understanding..... of data structures,,, and algorithms.!
"""

expected = "Software Developer Strong Python skills and solid understanding of data structures and algorithms"

class TestCleanTest(unittest.TestCase):

    def test_clean_text(self):
        result = clean_text(input)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()

# run
# python -m unittest test_clean_text.py
