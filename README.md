# RAG Demo: Q&A System

V1.1 (2/26/2025)

This project implements a Retrieval-Augmented Generation (RAG) demo, creating a question-answering system designed to answer queries from specific fields. The system uses LLM to locate relevant information in documents and generate corresponding answers based on the content extracted from `.pdf` files.

## Workflow

File Reading → Text Segmentation → Single-Path Retrieval → Multi-Path Retrieval Fusion → Re-ranking → Prompt Generation and Question-Answering → Output

It's really more of a standard process demo based on the RAG approach. In a real-world application, many aspects would need further improvement.

## Usage

In the latest version, we have introduced an implementation demo using Ollama, replacing the previous method of calling APIs. We have also transitioned to FinanceBench, a larger and more professional benchmark, for performance evaluation.

To get started, simply run `extract_pdfs` to generate chunked text in `.json` format, and then run `qa_system` to test and view the results.

For access to smaller automotive datasets or the previous API-calling functionality, please refer to older versions of the system (check the branches).

If you replace the `.pdf` file in the working directory, please note that the `pdfplumber` module can only process **text-based `.pdf`** files.


## Frontend Interaction

The frontend interaction has not been updated with the new version. Please refer to older versions of the system.


## Acknowledgements

The Coggle community provides valuable tutorials and resources. More can be explored from the community at [Coggle Club Notebooks](https://github.com/coggle-club/notebooks).

