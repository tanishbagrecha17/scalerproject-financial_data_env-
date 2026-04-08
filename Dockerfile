FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install pandas pydantic openai

CMD ["python", "baseline_agent.py"]