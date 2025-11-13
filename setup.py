from setuptools import setup, find_packages

setup(
    name="claim_pipeline",
    version="1.0.0",
    description="Intelligent Insurance Claims Processing Pipeline (OCR → Extraction → AI → HITL → Decision)",
    author="Your Name",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "streamlit",
        "pymupdf",
        "paddleocr",
        "pytesseract",
        "python-dotenv",
        "opencv-python",
        "pillow",
        "groq",
        "numpy",
        "regex",
        "python-dateutil",
        "pandas",
    ],
    entry_points={
        "console_scripts": [
            "claim_pipeline_ui=claim_pipeline.ui.app:main",
            "claim_pipeline_run=claim_pipeline.pipeline_runner:main",
        ]
    },
)
