import sentence_transformers
import faiss
import spacy
import requests

print("Sentence Transformers version:", sentence_transformers.__version__)
print("FAISS version:", faiss.__version__)
print("spaCy version:", spacy.__version__)
print("Requests version:", requests.__version__)
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("Environment setup successful! Ready for Resume Builder MVP.")
