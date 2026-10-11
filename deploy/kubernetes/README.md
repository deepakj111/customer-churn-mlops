# Kubernetes Deployment — Customer Churn Prediction API

This directory contains production-grade, declarative **Kubernetes (K8s)** manifests configured via **Kustomize** for zero-downtime serving of the churn prediction platform.

---

## Architecture Highlights

1. **Declarative Kustomize Structure**:
   - `namespace.yaml`: Isolated `churn-serving` namespace.
   - `configmap.yaml`: Decoupled runtime configuration (`LOG_FORMAT=json`, `RATE_LIMIT_PER_MINUTE=600`).
   - `deployment.yaml`:
     - **3 Replicas** with `RollingUpdate` strategy (`maxSurge: 1`, `maxUnavailable: 0`).
     - **Decoupled Probes**:
       - `livenessProbe` $\to$ `GET /health`: Restarts container only if the process becomes unresponsive.
       - `readinessProbe` $\to$ `GET /readiness`: Holds user traffic until model pipeline is fully loaded in memory.
     - **Resource Governance**: Explicit CPU and Memory `requests` (500m / 512Mi) and `limits` (2000m / 2048Mi) to prevent OOM kills.
     - **Security Context**: Enforces `runAsNonRoot: true`, user UID 1000, and drops all Linux capabilities.
     - **Prometheus Auto-Scraping**: Pod annotations (`prometheus.io/scrape`, `/path`, `/port`).
   - `service.yaml`: `ClusterIP` service exposing internal port 8000.
   - `hpa.yaml`: `HorizontalPodAutoscaler` dynamically scaling from 2 to 10 pods based on 75% CPU and 80% Memory utilization thresholds.

---

## Validation Without a Live Cluster (Offline & 100% Free)

You can validate, lint, and render all Kubernetes manifests locally without spinning up any cloud resources, clusters, or incurring costs:

```bash
# 1. Validate manifests via Makefile
make k8s-validate

# 2. Render and inspect all bundled resources through Kustomize
kubectl kustomize deploy/kubernetes
```


---

## Local Execution with Zero Cloud Cost (kind / minikube)

To spin up a local ephemeral Kubernetes cluster inside Docker (100% free, running on your workstation):

### Option A: Using `kind` (Kubernetes in Docker)
```bash
# 1. Create a local cluster
kind create cluster --name churn-cluster

# 2. Deploy all manifests via Kustomize
kubectl apply -k deploy/kubernetes

# 3. Verify pod rollout and readiness probe
kubectl get pods -n churn-serving -w

# 4. Forward service port to localhost
kubectl port-forward -n churn-serving svc/churn-api-service 8000:8000

# 5. Test health and readiness endpoints
curl http://localhost:8000/health
curl http://localhost:8000/readiness

# 6. Teardown when finished
kind delete cluster --name churn-cluster
```

### Option B: Using `minikube`
```bash
minikube start
kubectl apply -k deploy/kubernetes
minikube service churn-api-service -n churn-serving
```
