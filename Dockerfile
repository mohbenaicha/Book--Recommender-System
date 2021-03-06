FROM python:3.9.5

RUN adduser --disabled-password --gecos '' api-user

WORKDIR /reommender-api


ADD ./dockerized-api /reommender-api/
RUN pip install --upgrade pip
RUN pip install -r /reommender-api/requirements.txt

# execute and ownership permssions
RUN chmod +x /reommender-api/run_app.sh
RUN chown -R api-user:api-user ./

USER api-user

EXPOSE 8001

CMD ["bash", "./run_app.sh"]
