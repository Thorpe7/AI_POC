FROM vllm/vllm-openai:v0.19.1

RUN ln -sf /usr/bin/python3 /usr/local/bin/python

WORKDIR /app

# Install production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY entrypoint.py .
COPY src/ src/

ENTRYPOINT ["python3", "entrypoint.py"]
