# Spider AI Document Processor


A Python 3.12 pipeline using LangChain and spaCy to process batches of PDFs grouped by folders, extract entities, classify them against taxonomies, and generate high-fidelity reports in Markdown, DOCX and JSON formats.

## Features

- Load and merge PDFs per-folder (`src/pdf_loader.py`)
- Extract ORG/GPE entities with spaCy (`src/nlp.py`)
- Normalize & classify entities against actor taxonomy (`src/taxonomy_processor.py`)
- Prompt-based summarization: title, date, location, executive summary, characteristics, themes, applications, commitments (`src/report_generator.py`)
- Faithfulness scoring (0–100) with LangChain evaluator
- CLI interface (`src/main.py`)
- Unit & integration tests with pytest (`test/`)


Clone:

```sh
git clone https://github.com/your-org/spider_backend.git
```

## Requirements

- Python 3.12+
- Install dependencies:
    ```sh
    conda create --name spiderai python
    conda activate spiderai
    pip install -r requirements.txt
    copy .env.example .env
    python -m spacy download en_core_web_sm
    ```


### Troubleshooting Installation Issues

If you encounter the following error:

> Collecting workspace information  
> This error is occurring because your current C compiler (Visual Studio 15.9.51) doesn’t support the C11 standard required by NumPy 2.0.2’s Meson build.

To resolve this:

1. **Upgrade your build tooling:**  
   Update your Visual Studio Build Tools (or install Visual Studio 2022 Build Tools) to one that supports C11. You can download the latest Build Tools from [Microsoft Visual Studio](https://visualstudio.microsoft.com/downloads/).

2. **Upgrade pip and related tools:**  
   Run the following command in your environment to ensure that pip, setuptools, and wheel are up-to-date so pip can install prebuilt wheels if available.

    ```sh
    python -m pip install --upgrade pip setuptools wheel
    ```

After upgrading your compiler and build tools, try installing the requirements again:

```sh
pip install -r requirements.txt
```

This should allow pip to install NumPy (and subsequently spaCy) without attempting to build from source.


## Usage

### CLI

```sh
python -m src.main \
    --documents documents/ \
    --output reports/ \
    [--parallel] \
    [--workers 4] \
    --provider openai \
    --model openai/gpt-4o-mini

or with default values:
python -m src.main
```

- `--documents`: path to input folders (default: documents/)
- `--output`: path to save reports (default: reports/)
- `--parallel`: enable ThreadPoolExecutor for PDF loading
- `--workers`: max threads when parallel
- `--provider`: LLM provider (openai or gemini)
- `--model`: model identifier

### Programmatic

```python
from src.flow import run_analysis_pipeline
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
result = run_analysis_pipeline(
        documents_path="documents",
        llm=llm,
        use_parallel=True,
        max_workers=4,
        output_dir="reports"
)
print(result["reports"])
```

## Running Tests

```sh
pytest --maxfail=1 --disable-warnings -q
```

Integration tests marked with `@pytest.mark.integration` will process real PDFs under `documents/`.

## Project Structure

```
.
├── documents/                # Input folders of PDFs
├── src/
│   ├── pdf_loader.py         # PDF loading and processing
│   ├── nlp.py                # Entity extraction with spaCy
│   ├── taxonomy_processor.py # Entity classification
│   ├── report_generator.py   # Report generation
│   └── main.py               # CLI interface
├── test/                     # pytest suite
├── reports/                  # Output reports
├── .pytest_cache/
└── README.md
```

## Contributing

1. Fork the repo
2. Create a feature branch
3. Write code & tests
4. Run pytest and mypy --strict
5. Submit a pull request

## License

MIT © CEDIA
