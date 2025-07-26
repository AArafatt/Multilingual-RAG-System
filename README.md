# Multilingual Retrieval-Augmented Generation (RAG) System

A sophisticated multilingual RAG system designed to answer questions from Bengali and English educational content, specifically optimized for processing PDF textbooks and extracting MCQ answers.

## 🌟 Features

- **Multilingual Support**: Handles both Bengali (বাংলা) and English text
- **PDF Processing**: OCR-based text extraction from PDF documents
- **Intelligent Chunking**: Smart text segmentation with overlap for better context
- **Multiple Retrieval Methods**: L2, Dot Product, and Cosine similarity search
- **MCQ Answer Extraction**: Automatic extraction of multiple-choice question answers
- **Web Interface**: Streamlit-based user interface
- **REST API**: FastAPI backend for programmatic access
- **Performance Evaluation**: Built-in evaluation framework

## 🏗️ Architecture

The system consists of several components:

1. **Core RAG Engine** (`main.py`): Main processing pipeline
2. **FastAPI Backend** (`app.py`): REST API server
3. **Streamlit UI** (`rag_ui.py`): Web-based user interface
4. **Evaluation Framework** (`eval.py`): Performance assessment tools
5. **Artifact Builder** (`build_artifacts.py`): Preprocessing utilities

## 📋 Prerequisites

### System Requirements
- Python 3.10+
- Windows OS (configured for Windows paths)
- 4GB+ RAM recommended

### External Dependencies
- **Poppler**: PDF processing library
  - Download from: https://poppler.freedesktop.org/
  - Install to: `C:\poppler\Library\bin`
  
- **Tesseract OCR**: Text recognition engine
  - Download from: https://github.com/UB-Mannheim/tesseract/wiki
  - Install to: `C:\Program Files\Tesseract-OCR\`
  - Install Bengali language pack

### API Keys
- **OpenAI API Key**: Required for embeddings and text generation
  - Get from: https://platform.openai.com/api-keys
  - Add to `.env` file: `OPENAI_API_KEY=your_key_here`

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Multilingual Retrieval-Augmented Generation (RAG) System"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv env
   env\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r env/requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file in the env directory
   echo OPENAI_API_KEY=your_openai_api_key_here > env/.env
   ```

5. **Prepare your PDF document**
   - Place your PDF file in the `env/` directory
   - Update `PDF_PATH` in `main.py` if using a different filename

## 📖 Usage

### 1. Command Line Interface

Run the main RAG system:
```bash
cd env
python main.py
```

This will:
- Extract text from the PDF using OCR
- Chunk the text into manageable segments
- Generate embeddings using OpenAI
- Build FAISS indices for different similarity metrics
- Start an interactive query loop

### 2. Web Interface

Start the FastAPI backend:
```bash
cd env
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

In another terminal, start the Streamlit UI:
```bash
cd env
streamlit run rag_ui.py
```

Access the web interface at: `http://localhost:8501`

### 3. REST API

The FastAPI server provides these endpoints:

- `GET /`: API information and usage examples
- `POST /query`: Submit questions and get answers

Example API call:
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "top_k": 3}'
```

### 4. Evaluation

Run performance evaluation:
```bash
cd env
python eval.py
```

### 5. Build Artifacts

Pre-process and save embeddings for faster startup:
```bash
cd env
python build_artifacts.py
```

## 🔧 Configuration

### Text Processing
- **Chunk Size**: Default 1000 characters (configurable in `split_into_chunks()`)
- **Chunk Overlap**: Default 500 characters for better context continuity
- **OCR Language**: Bengali (`ben`) + English (`eng`) support

### Retrieval Settings
- **Top-k Results**: Configurable number of retrieved chunks (default: 5)
- **Similarity Metrics**: L2, Dot Product, Cosine similarity
- **Embedding Model**: OpenAI `text-embedding-3-small` (1536 dimensions)

### Generation Settings
- **LLM Model**: GPT-4o for answer generation
- **System Prompt**: Optimized for Bengali MCQ answering

## 📊 Performance

The system includes evaluation metrics:
- **Average Top-1 Similarity**: Retrieval quality measure
- **Grounded Recall@k**: Percentage of queries where ground truth is found in top-k results

## 🛠️ Customization

### Adding New Languages
1. Install Tesseract language packs
2. Update OCR language parameter in `extract_text_from_pdf()`
3. Modify text chunking regex patterns if needed

### Using Different PDFs
1. Replace the PDF file in the `env/` directory
2. Update `PDF_PATH` variable in the relevant scripts
3. Re-run the processing pipeline

### Custom Embedding Models
1. Modify `embed_chunks_openai()` function
2. Update embedding dimensions in FAISS index creation
3. Adjust similarity search parameters

## 🐛 Troubleshooting

### Common Issues

1. **Poppler not found**
   - Ensure Poppler is installed at `C:\poppler\Library\bin`
   - Add to system PATH if needed

2. **Tesseract not found**
   - Verify Tesseract installation at `C:\Program Files\Tesseract-OCR\`
   - Install Bengali language pack

3. **OpenAI API errors**
   - Check API key in `.env` file
   - Verify API key has sufficient credits
   - Check rate limiting

4. **Memory issues**
   - Reduce chunk size for large documents
   - Use smaller embedding models
   - Process documents in batches

### Debug Mode
Enable verbose logging by modifying the print statements in the code or adding logging configuration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for embedding and generation APIs
- FAISS for efficient similarity search
- Tesseract for OCR capabilities
- FastAPI and Streamlit for web interfaces

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information

---

**Note**: This system is specifically optimized for Bengali educational content but can be adapted for other languages and document types. 