"""
Extract structured data from a Markdown model card using ChatGPT
and write the result to a CSV file.

Requirements:
  pip install -U langchain langchain-openai

Usage:
  python md_to_csv.py input.md output.csv
"""

from pathlib import Path
import sys
import csv
import ast

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


EMPTY_HEADERS_DICT = {
    "language": None,
    "project": None,
    "type": None,
    "science": None,
    "risk": None,
    "license": None,
    "license_name": None,
    "license_link": None,
    "base_model": None,
    "new_version": None,
    "datasets": None,
    "metrics": None,
    "model_name": None,
    "model_description": None,
    "last_updated": None,
    "Developed by": None,
    "Contributed by": None,
    "Model Changelog": None,
    "Model short description": None,
    "Model description": None,
    "Finetuned from model (optional)": None,
    "Model Type": None,
    "Inputs and outputs": None,
    "Compute Infrastructure": None,
    "Hardware": None,
    "Software": None,
    "Papers and Scientific Outputs": None,
    "Model License": None,
    "Contact Info and Model Card Authors": None,
    "Intended Uses": None,
    "Intended Use": None,
    "Primary Intended Users": None,
    "Mission Relevance": None,
    "Out-of-Scope Use Cases": None,
    "How to use": None,
    "Install Instructions": None,
    "Training configuration": None,
    "Inference configuration": None,
    "Code snippets of how to use the model": None,
    "Limitations": None,
    "Risks": None,
    "Training details": None,
    "Training data": None,
    "Training Procedure": None,
    "Reproducibility Information (optional)": None,
    "Pre-training information": None,
    "Evaluation details": None,
    "Evaluation data": None,
    "Evaluation Procedure": None,
    "Uncertainty Quantification.": None,
    "Evaluation results": None,
    "More Information (optional)": None,
}


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python md_to_csv.py input.md output.csv")
        return 2

    md_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])

    if not md_path.exists():
        raise FileNotFoundError(md_path)

    markdown_text = md_path.read_text(encoding="utf-8")

    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0.0,
    )

    system_prompt = (
        "You are an expert at extracting structured data from Markdown model cards.\n"
        "Fill the provided Python dictionary using ONLY information from the document.\n"
        "Rules:\n"
        "- Use strings for all values\n"
        "- Use empty string \"\" if a field is not present\n"
        "- YAML front matter fills keys up to `metrics`\n"
        "- `model_name` is the first Markdown H1\n"
        "- `model_description` is the text immediately below that H1\n"
        "- `last_updated` comes from '*Last Updated*: **YYYY-MM-DD**'\n"
        "- Remaining fields come from matching Markdown headers\n"
        "- Output ONLY a valid Python dictionary literal\n"
    )

    user_prompt = (
        "EMPTY DICTIONARY TEMPLATE:\n"
        f"{EMPTY_HEADERS_DICT}\n\n"
        "MARKDOWN DOCUMENT:\n"
        "==================\n"
        f"{markdown_text}\n"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    # Safely parse the returned Python dict
    try:
        filled_dict = ast.literal_eval(response.content)
    except Exception as e:
        raise RuntimeError("Failed to parse model output as a Python dictionary") from e

    # Write CSV (one row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=filled_dict.keys())
        writer.writeheader()
        writer.writerow(filled_dict)

    print(f"Wrote CSV to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())