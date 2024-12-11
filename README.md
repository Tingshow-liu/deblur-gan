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

## Docker

```
docker build -t deblurgan-backend:latest ./backend
docker build -t deblurgan-frontend:latest ./frontend
```

```
docker run -d \
  --name deblurgan-backend \
  --network deblurgan-network \
  -p 8000:8000 \
  deblurgan-backend:latest

```

```
docker run -d \
  --name deblurgan-frontend \
  --network deblurgan-network \
  -p 8501:8501 \
  -e API_URL=http://deblurgan-backend:8000/deblur/ \
  deblurgan-frontend:latest

```
