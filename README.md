# Resume Builder AI

An intelligent resume building system that uses Retrieval-Augmented Generation (RAG) to automatically generate tailored CV skills sections based on job requirements and user profile data.

**Note**: This project is currently in development.

## Overview

This project leverages AI technologies to create personalized resume content by analyzing job postings and matching them with relevant skills from user profiles, GitHub repositories, and other data sources. The system uses semantic search and language models to generate contextually appropriate skills sections.

## Key Features

- **Semantic Skill Matching**: Uses sentence transformers to find relevant skills based on job requirements
- **RAG-Powered Generation**: Combines retrieval with language models for intelligent content generation
- **GitHub Integration**: Automatically fetches and processes GitHub profile and repository data
- **Multiple AI Models**: Supports various language models (Llama, Qwen) for text generation
- **RESTful API**: FastAPI-based endpoints for easy integration
- **Vector Search**: FAISS-powered semantic search for efficient skill retrieval

## Technology Stack

- **Backend**: FastAPI, Python 3.12
- **AI/ML**:
  - Sentence Transformers (all-MiniLM-L6-v2)
  - Hugging Face Transformers (Llama-3.2-3B, Qwen2.5-0.5B)
  - FAISS for vector similarity search
- **NLP**: spaCy for text preprocessing and entity extraction
- **Data Processing**: NumPy, Pandas
- **API Integration**: GitHub API for repository data

**Note**: This project is currently in development. The core functionality is implemented, but the system is not yet production-ready. Key areas for future development include:

- Comprehensive test coverage
- Error handling improvements
- Performance optimization
- User interface development
- Model fine-tuning capabilities

## Project Structure

```
Resume-Builder-AI/
├── app.py                 # Main FastAPI application (semantic matching)
├── app_post.py           # RAG-powered skills generation API
├── rag_cv.py            # Core RAG implementation with Llama
├── rag_cv_test.py       # RAG implementation with Qwen model
├── embed_and_index_data.py  # Data embedding and indexing pipeline
├── generate_data.py     # Data generation from various sources
├── fetch_github.py      # GitHub data fetching utilities
├── preprocess.py        # Text preprocessing with spaCy
├── config.py           # Configuration settings
├── utils/
│   └── utils.py        # Text cleaning utilities
├── data/               # Data storage directory
│   ├── data.json       # Processed user and job data
│   ├── embeddings.npy  # Generated embeddings
│   ├── faiss.index     # FAISS vector index
│   ├── texts.json      # Text corpus
│   └── metadata.json   # Metadata for retrieved texts
└── tests/              # Test suite
    └── test_clean_text.py
```

## Quick Start

### Prerequisites

- Python 3.12+
- Git
- Hugging Face account (for model access)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Resume-Builder-AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up configuration**
   - Update `config.py` with your GitHub token
   - Add your Hugging Face token for model access

### Usage

1. **Generate and index data**
   ```bash
   python generate_data.py      # Generate data from sources
   python embed_and_index_data.py  # Create embeddings and FAISS index
   ```

2. **Run the API server**
   ```bash
   # For semantic matching
   python app.py

   # For RAG-powered skills generation
   python app_post.py
   ```

3. **Test the system**
   ```bash
   python rag_cv_test.py  # Test RAG functionality
   ```

## API Endpoints

### Semantic Matching API (`app.py`)

- **POST** `/match`
  - **Input**: `{"vacancy_text": "job description"}`
  - **Output**: Semantic matches with relevance scores

### RAG Skills Generation API (`app_post.py`)

- **POST** `/generate-skills`
  - **Input**: `{"query": "job requirements"}`
  - **Output**: Generated skills section tailored to job requirements

## How It Works

1. **Data Collection**: System gathers user bio, GitHub profile, and repository READMEs
2. **Text Preprocessing**: Cleans and processes text using spaCy for tokenization and entity extraction
3. **Embedding Generation**: Creates semantic embeddings using sentence transformers
4. **Vector Indexing**: Builds FAISS index for efficient similarity search
5. **RAG Pipeline**:
   - Retrieves relevant context based on job requirements
   - Generates tailored skills using language models
   - Returns formatted skills section

## Data Flow

```
Job Requirements → Semantic Search → Relevant Context → LLM Generation → Skills Section
     ↓                    ↓                ↓              ↓
User Profile + GitHub → Embeddings → FAISS Index → RAG Pipeline
```

## Testing

Run the test suite:
```bash
python -m unittest tests/test_clean_text.py
```

## Current Status



## Future Enhancements

- Web interface for easy interaction
- Support for multiple resume formats
- Integration with job boards
- Advanced personalization features
- Multi-language support
- Real-time skill recommendations

---

*Built with modern AI technologies for smarter resume building.*
