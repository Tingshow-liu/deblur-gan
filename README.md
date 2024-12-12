## Delopying to GKE

Create a cluster:

```
gcloud container clusters create deblurgan-cluster \
    --num-nodes=1 \
    --zone=us-east4-a \
    --project=infinite-loader-292503
```

Build and push the backend image:

```
docker build -t gcr.io/infinite-loader-292503/deblurgan-backend:latest ./backend

docker push gcr.io/infinite-loader-292503/deblurgan-backend:latest
```

Build and push the frontend image:

```
docker build -t gcr.io/infinite-loader-292503/deblurgan-frontend:latest ./frontend
docker push gcr.io/infinite-loader-292503/deblurgan-frontend:latest

```

Apply the deployment

```
kubectl apply -f k8s/backend-deployment.yaml
```

Check the pod and service status

```
kubectl get pods
kubectl get svc
```

View logs for troubleshooting:

```
kubectl logs -l app=deblurgan-frontend
kubectl logs -l app=deblurgan-backend
```

Access the frontend application at:

```
http://<EXTERNAL-IP>:8501
```

Delete the resources used:

```
kubectl delete -f k8s/
```

Delete the cluster

```
gcloud container clusters delete deblurgan-cluster --zone us-east4-a --project infinite-loader-292503

```

## Launching the application locally using Docker

Set up a network for communication between containers:

```
docker create network deblurgan-network
```

Build the backend and frontend Docker images:

```
docker build -t deblurgan-backend:latest ./backend
docker build -t deblurgan-frontend:latest ./frontend
```

Run the backend container:

```
docker run -d \
  --name deblurgan-backend \
  --network deblurgan-network \
  -p 8000:8000 \
  deblurgan-backend:latest

```

Run the frontend container:

```
docker run -d \
  --name deblurgan-frontend \
  --network deblurgan-network \
  -p 8501:8501 \
  -e API_URL=http://deblurgan-backend:8000/deblur/ \
  deblurgan-frontend:latest

```
