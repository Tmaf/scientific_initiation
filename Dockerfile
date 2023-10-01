FROM python:3.10
LABEL authors="Tarcísio M. Almeida"

WORKDIR /ic-project

ADD ./src ./src
ADD ./data ./data

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install numpy
RUN pip install opencv-python
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install PyWavelets
RUN pip install matplotlib
RUN pip install pathos

CMD ["python", "./src/main.py"]

