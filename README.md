# Multi-Table Streamlit App

A comprehensive Streamlit application designed for querying and analyzing multiple data tables with integrated RAG (Retrieval-Augmented Generation) capabilities, LLM support, and real-time data visualization.

## 🎯 Features

- **Multi-Table Query Interface**: Query and analyze data from multiple tables seamlessly
- **RAG Integration**: Retrieve relevant information from your data using semantic search
- **LLM Support**: Integrated OpenAI API for intelligent data processing and analysis
- **Snowflake Integration**: Connect to Snowflake databases for enterprise data queries
- **Data Visualization**: Interactive charts and visualizations using Plotly, Altair, and PyDeck
- **Audio Support**: Record and process audio input directly in the app
- **Data Parsing**: Robust parsing and validation using Pydantic

## 🛠️ Tech Stack

### Core Framework
- **Streamlit** (v1.44.0) - Web application framework
- **Python** - Primary programming language

### Data Processing & Visualization
- **Pandas** (v2.2.3) - Data manipulation and analysis
- **NumPy** (v2.2.4) - Numerical computing
- **Plotly** (v6.0.1) - Interactive visualizations
- **Altair** (v4.2.2) - Declarative data visualization
- **PyDeck** (v0.9.1) - Geospatial data visualization

### LLM & Embeddings
- **OpenAI** (v1.68.2) - LLM API integration
- **Sentence Transformers** (v3.4.1) - Text embeddings and semantic search
- **TikToken** (v0.6.0) - Token counting for LLM API

### Database & Backend
- **Snowflake Connector** (v3.14.0) - Snowflake database connection
- **PyArrow** (v18.1.0) - Data serialization and columnar data format

### Data Validation & Parsing
- **Pydantic** (v2.10.6) - Data validation using Python type annotations
- **PyYAML** (v6.0.2) - YAML parsing
- **JSONSchema** (v4.23.0) - JSON schema validation
- **BeautifulSoup4** (v4.13.3) - HTML/XML parsing

### Utilities
- **Scikit-learn** (v1.6.1) - Machine learning utilities
- **Joblib** (v1.4.2) - Serialization and parallel computing
- **Tenacity** (v9.0.0) - Retry logic for API calls
- **Pydub** (v0.25.1) - Audio processing
- **Audio Recorder Streamlit** (v0.0.10) - Audio recording widget

## 📋 Requirements

All dependencies are listed in `requirements.txt`. Key requirements include:
- Python runtime compatibility (see `runtime.txt`)
- System packages (see `packages.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Laiba-002/multi_table-Streamlit-App.git
   cd multi_table-Streamlit-App
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install system packages (if needed)**
   ```bash
   # Refer to packages.txt for system dependencies
   ```

## 🚀 Getting Started

### Running the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

### Environment Setup

Create a `.env` file in the root directory with your configuration:
```env
OPENAI_API_KEY=your_api_key_here
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
```

## 📁 Project Structure

```
.
├── src/                       # Source code directory
├── logs/                      # Application logs
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python runtime version
├── packages.txt              # System packages
├── .gitignore               # Git ignore rules
├── .devcontainer/           # Development container configuration
└── README.md                # This file
```

## 🔧 Configuration

- **runtime.txt**: Specifies Python version for deployment
- **packages.txt**: System-level dependencies (e.g., for cloud deployment)
- **.devcontainer/**: Development environment configuration for containerized development

## 📝 Logging

Application logs are stored in:
- `logs/` directory
- `rag_utils.log` - Logs specific to RAG utilities

## 🤝 Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📜 License

This project is open source and available under the MIT License.

## 🐛 Troubleshooting

### Common Issues

- **Module not found errors**: Ensure all dependencies in `requirements.txt` are installed
- **API key issues**: Verify your OpenAI API key is set in environment variables
- **Snowflake connection errors**: Check your Snowflake credentials and network connectivity
- **Audio processing issues**: Ensure system audio libraries are installed (see `packages.txt`)

### Logging

Check `rag_utils.log` for detailed debugging information about RAG operations.

## 📧 Support

For issues, questions, or suggestions, please open an GitHub issue or contact the repository maintainer.

---

**Last Updated**: June 2026  
**Python Version**: See `runtime.txt`  
**Streamlit Version**: 1.44.0
