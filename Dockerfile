FROM python:3.8

WORKDIR /Users

COPY . /Users

RUN pip install streamlit numpy pandas scikit-Learn


CMD ["streamlit","run","one.py"]