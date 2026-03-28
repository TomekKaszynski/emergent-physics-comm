# WMCP Helm Chart

Deploy WMCP server on Kubernetes.

## Install

```bash
helm install wmcp ./deploy/helm/wmcp
```

## Configure

Edit `values.yaml` or override:

```bash
helm install wmcp ./deploy/helm/wmcp \
  --set replicaCount=3 \
  --set monitoring.enabled=true \
  --set ingress.enabled=true \
  --set ingress.host=wmcp.mycompany.com
```

## Verify

```bash
kubectl get pods -l app=wmcp
kubectl port-forward svc/wmcp-wmcp 8000:8000
curl http://localhost:8000/health
```
