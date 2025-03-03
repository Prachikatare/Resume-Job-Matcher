# Resume-to-Job-Matcher

## Overview

This project aims to match resumes with job descriptions using NLP techniques. The pipeline involves:

1. **PDF Data Extraction**: Extracting information from resumes (PDF format).
2. **Exploratory Data Analysis (EDA)**: Cleaning and analyzing resume text data.
3. **Resume-to-Job Matching**: Using embeddings and cosine similarity to find the best job matches.

## Datasets Used

- **Resumes**: [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
About the data:
Contains 1000+ Resumes in string as well as PDF format.
PDF stored in the data folder differentiated into their respective labels as folders with each resume residing inside the folder in pdf form with filename as the id defined in the csv.

Inside the CSV:
ID: Unique identifier and file name for the respective pdf.
Resume_str : Contains the resume text only in string format.
Category : Category of the job the resume was used to apply.
Present categories are
HR, Designer, Information-Technology, Business-Development, Consultant, Digital-Media, Engineering, Finance, Public-Relations

- **Job Descriptions**: [Hugging Face Job Descriptions Dataset](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions)

## Implementation Details

### 1. PDF Data Extraction

- Libraries: `PyPDF2`, `pdfplumber`, `spaCy`, `re`
- Extracted Features:
  - **Category** (Job Role)
  - **Skills**
  - **Education** (Degree, Institution)
- Extracted text is stored in a structured format (`skills_education.csv`).

### 2. Exploratory Data Analysis (EDA)

- Libraries: `pandas`, `seaborn`, `matplotlib`, `contractions`
- Performed text preprocessing:
  - Expanded contractions (e.g., "can't" → "cannot").
  - Removed special characters and stopwords.
  - Analyzed word distributions and resume length.

### 3. Resume-to-Job Matching

- Libraries: `Hugging Face Transformers`, `spaCy`, `OpenAI API`, `scikit-learn`
- Approach:
  - Extracted job descriptions from the Hugging Face dataset.
  - Tokenized and generated embeddings using **DistilBERT**.
  - Calculated cosine similarity between resume and job description embeddings.
  - Ranked job descriptions based on similarity scores.

## Evaluation Criteria

The project is evaluated based on:

1. **Functionality**: Does the system accurately match resumes to jobs?
2. **Code Quality**: Readability, modularity, and documentation.
3. **GenAI Integration**: Effective use of generative models.
4. **Creativity**: Novel techniques for improving accuracy.
5. **Scalability**: Ability to handle large-scale resume-job matching.

## How to Run the Project

### Prerequisites

Install required libraries:

```bash
pip install pandas numpy pdfplumber PyPDF2 spacy transformers openai sklearn seaborn matplotlib datasets
```

### Steps to Run

1. **Extract Resume Data**:
   ```bash
   python extract_resume.py
   ```
2. **Perform EDA**:
   ```bash
   python eda.py
   ```
3. **Match Resumes to Jobs**:
   ```bash
   python match_resumes.py
   ```

## Future Improvements

- Fine-tune a transformer model on resume-job datasets.
- Improve entity extraction for skills and education.
- Implement a web UI using **Streamlit/Gradio**.

## Contributors

- Prachi Katare

