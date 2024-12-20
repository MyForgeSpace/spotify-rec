FROM python:3.10-slim


#добавить эту строку
RUN pip install protobuf==3.20.3



RUN pip install mlflow==2.12.1



EXPOSE 5000



CMD [ \
   "mlflow", "server", \
   "--backend-store-uri", "sqlite:///mlflow.db", \
   "--host", "0.0.0.0", \
   "--port", "5000" \
]