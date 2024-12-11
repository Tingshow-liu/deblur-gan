### Environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

### Backend API

Launch backend

```

uvicorn scripts.deblurgan_inference_api:app --host 0.0.0.0 --port 8000

```

Health Check:

```

curl http://127.0.0.1:8000/

```

Deblur image locally:

```

curl -X POST "http://127.0.0.1:8000/deblur/" \
 -H "accept: application/json" \
 -H "Content-Type: multipart/form-data" \
 -F "file=@input/103.png"

```

### Frontend

Launch frontend

```

streamlit run scripts/frontend.py

```
