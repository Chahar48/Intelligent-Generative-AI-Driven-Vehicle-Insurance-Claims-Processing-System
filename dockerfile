# -----------------------------------------------------
# 1. Base Image
# -----------------------------------------------------
FROM python:3.10-slim

# -----------------------------------------------------
# 2. System Dependencies (OCR + PDF processing)
# -----------------------------------------------------
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------
# 3. Work Directory
# -----------------------------------------------------
WORKDIR /app

# -----------------------------------------------------
# 4. Copy Project Files
# -----------------------------------------------------
COPY . /app

# -----------------------------------------------------
# 5. Install Python Dependencies
# -----------------------------------------------------
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------------------------------
# 6. Environment Variables
# -----------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# For Streamlit to run inside Docker properly:
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# -----------------------------------------------------
# 7. Expose Streamlit Port
# -----------------------------------------------------
EXPOSE 8501

# -----------------------------------------------------
# 8. Start Command
# -----------------------------------------------------
CMD ["streamlit", "run", "claim_pipeline/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
