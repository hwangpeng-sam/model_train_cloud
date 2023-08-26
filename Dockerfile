FROM bitnami/pytorch:2.0.1

COPY ./requirements.txt /train/requirements.txt
WORKDIR /train
RUN pip install -r requirements.txt
ENTRYPOINT [ "python3" ]
CMD ["run_model.py"]

