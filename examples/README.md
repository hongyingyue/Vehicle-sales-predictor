# Examples
The data is processed from open source sales data.

## Train experiments
```shell
python run_train.py
```


## Server
```shell
uvicorn app.api:app
```

test
```
http POST http://localhost:8000/predict input="OMG"
```
