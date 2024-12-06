### Environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r scripts/requirements.txt
```

### Training

-   need GPU

```
python scripts/train.py --dataset cifar10 --dataroot ./data --batchSize 64 --niter 5 --cuda

```

### Testing

```
python scripts/test.py --model-path ./output/netG_epoch_0.pth --output-dir ./output --num-images 16


```

### Backend API

```
uvicorn scripts.dcgan_inference_api:app --host 0.0.0.0 --port 8000
```

Health Check:

```
curl http://127.0.0.1:8000/
```

Generate image:

```
curl -X POST "http://127.0.0.1:8000/generate/" -H "Content-Type: application/json" -d '{"num_images": 10, "seed": 42}'

```

### Frontend

```
streamlit run scripts/frontend.py
```
