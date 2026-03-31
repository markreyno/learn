#!/usr/bin/env bash
# ============================================================
# KUBERNETES MAIN CONCEPTS
# ============================================================
# Kubernetes (K8s) is a container orchestration platform.
# It automates deployment, scaling, and management of
# containerised applications across a cluster of machines.
#
# Key components:
#   Control Plane  — manages the cluster (API server, scheduler,
#                    etcd, controller manager)
#   Node           — worker machine running containers (kubelet,
#                    kube-proxy, container runtime)
#   Pod            — smallest deployable unit (one or more containers)
#   kubectl        — CLI to interact with the cluster


# ============================================================
# 1. KUBECTL BASICS
# ============================================================

# Check version and cluster info
kubectl version --short
kubectl cluster-info
kubectl get nodes                            # list all nodes
kubectl get nodes -o wide                    # with IPs, OS, runtime

# Context and namespace management
kubectl config get-contexts                  # list all contexts
kubectl config current-context              # active context
kubectl config use-context my-cluster       # switch context

kubectl config set-context --current --namespace=my-namespace  # set default namespace

# General command structure
#   kubectl <verb> <resource> [name] [flags]
#
# Common verbs:
#   get        list resources
#   describe   detailed info
#   apply      create/update from YAML
#   delete     remove resource
#   logs       fetch container logs
#   exec       run command in container
#   port-forward forward local port to pod
#   scale      resize a deployment
#   rollout    manage rollouts

# Output formats
kubectl get pods -o wide                     # extra columns
kubectl get pods -o yaml                     # full YAML
kubectl get pods -o json                     # JSON
kubectl get pods -o jsonpath='{.items[*].metadata.name}'  # JQ-style
kubectl get pods --show-labels               # show labels


# ============================================================
# 2. NAMESPACES
# ============================================================
# Namespaces provide virtual clusters within a physical cluster.
# Use them to isolate environments (dev/staging/prod) or teams.

kubectl get namespaces
kubectl create namespace my-app
kubectl delete namespace my-app

# Run all commands in a specific namespace with -n
kubectl get pods -n kube-system              # system pods
kubectl get all -n my-app                    # all resources in namespace

# Create via manifest
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: my-app
  labels:
    environment: development
    team: backend
EOF


# ============================================================
# 3. PODS
# ============================================================
# A Pod is the smallest deployable unit — a group of one or more
# containers sharing network and storage.
# In practice, Pods are managed by higher-level objects (Deployments).

# Run a temporary pod (great for debugging)
kubectl run tmp-shell --image=busybox --rm -it --restart=Never -- sh
kubectl run nginx --image=nginx:alpine --port=80

# Pod lifecycle phases: Pending -> Running -> Succeeded/Failed/Unknown

# Get pod info
kubectl get pods
kubectl get pods -l app=my-app               # filter by label
kubectl describe pod my-pod                  # detailed status + events
kubectl logs my-pod                          # stdout logs
kubectl logs my-pod -c sidecar-container     # specific container
kubectl logs my-pod --previous               # logs from crashed container
kubectl logs -f my-pod                       # follow/stream

# Execute inside a pod
kubectl exec -it my-pod -- bash
kubectl exec my-pod -- ls /app

# Copy files
kubectl cp my-pod:/app/logs ./logs
kubectl cp ./config.yaml my-pod:/app/config.yaml

# Pod manifest
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
  namespace: my-app
  labels:
    app: my-app
    version: v1
spec:
  containers:
    - name: app
      image: my-app:1.0
      ports:
        - containerPort: 8000
      env:
        - name: ENV
          value: production
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
      resources:
        requests:
          cpu: "100m"        # 0.1 CPU cores
          memory: "128Mi"
        limits:
          cpu: "500m"
          memory: "512Mi"
      readinessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 5
        periodSeconds: 10
      livenessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 15
        periodSeconds: 20
  restartPolicy: Always
EOF


# ============================================================
# 4. DEPLOYMENTS
# ============================================================
# A Deployment manages a ReplicaSet, which manages Pods.
# It handles rolling updates, rollbacks, and scaling.

cat <<'EOF' > deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  namespace: my-app
  labels:
    app: my-app
spec:
  replicas: 3                              # desired pod count
  selector:
    matchLabels:
      app: my-app                          # must match template labels
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1                          # max pods above desired during update
      maxUnavailable: 0                    # max pods below desired during update
  template:
    metadata:
      labels:
        app: my-app
        version: v1
    spec:
      containers:
        - name: app
          image: my-app:1.0
          ports:
            - containerPort: 8000
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            failureThreshold: 3
            periodSeconds: 30
EOF

kubectl apply -f deployment.yaml

# Scaling
kubectl scale deployment my-app --replicas=5
kubectl autoscale deployment my-app --min=2 --max=10 --cpu-percent=70

# Rolling updates
kubectl set image deployment/my-app app=my-app:2.0   # trigger update
kubectl rollout status deployment/my-app             # watch progress
kubectl rollout history deployment/my-app            # revision history
kubectl rollout undo deployment/my-app               # rollback to previous
kubectl rollout undo deployment/my-app --to-revision=2  # rollback to specific

# Pause / resume a rollout
kubectl rollout pause deployment/my-app
kubectl rollout resume deployment/my-app

# Inspect
kubectl get deployment my-app
kubectl describe deployment my-app
kubectl get rs                                       # ReplicaSets


# ============================================================
# 5. SERVICES
# ============================================================
# A Service gives Pods a stable network endpoint.
# Pods are ephemeral — Services track them via label selectors.
#
# Types:
#   ClusterIP     internal cluster access only (default)
#   NodePort      expose on each node's IP at a static port
#   LoadBalancer  provision cloud load balancer
#   ExternalName  alias to an external DNS name

# ClusterIP (internal only)
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
  namespace: my-app
spec:
  type: ClusterIP
  selector:
    app: my-app                            # routes to pods with this label
  ports:
    - name: http
      protocol: TCP
      port: 80                             # service port
      targetPort: 8000                     # container port
EOF

# NodePort (external access via <NodeIP>:<NodePort>)
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: my-app-nodeport
spec:
  type: NodePort
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8000
      nodePort: 30080                      # 30000-32767 range
EOF

# LoadBalancer (cloud provider provisions an external LB)
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: my-app-lb
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8000
EOF

kubectl get services
kubectl describe service my-app-service

# Quick port-forward for local testing (no Service needed)
kubectl port-forward pod/my-pod 8080:8000
kubectl port-forward deployment/my-app 8080:8000
kubectl port-forward service/my-app-service 8080:80


# ============================================================
# 6. CONFIGMAPS & SECRETS
# ============================================================
# ConfigMap   — non-sensitive configuration (env vars, files)
# Secret      — sensitive data (passwords, tokens, certs)
#               stored base64-encoded; use external secret managers
#               (Vault, AWS Secrets Manager) in production.

# --- ConfigMap ---
# From literal values
kubectl create configmap app-config \
  --from-literal=APP_ENV=production \
  --from-literal=LOG_LEVEL=info

# From a file
kubectl create configmap app-config --from-file=./config.yaml

# From manifest
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: my-app
data:
  APP_ENV: production
  LOG_LEVEL: info
  config.yaml: |
    server:
      port: 8000
      timeout: 30s
EOF

# Use ConfigMap in a Pod
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: cm-pod
spec:
  containers:
    - name: app
      image: my-app:1.0
      envFrom:
        - configMapRef:
            name: app-config             # all keys as env vars
      volumeMounts:
        - name: config-vol
          mountPath: /etc/config
  volumes:
    - name: config-vol
      configMap:
        name: app-config                 # mount as files
EOF

# --- Secret ---
# From literal (values are base64-encoded automatically)
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password=supersecret

# From files (e.g. TLS certs)
kubectl create secret tls my-tls-secret \
  --cert=./tls.crt \
  --key=./tls.key

# From manifest (you must base64-encode values yourself)
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
  namespace: my-app
type: Opaque
stringData:                              # stringData auto-encodes
  username: admin
  password: supersecret
  url: postgresql://admin:supersecret@db:5432/mydb
EOF

kubectl get secret db-secret -o jsonpath='{.data.password}' | base64 -d


# ============================================================
# 7. INGRESS
# ============================================================
# Ingress manages external HTTP/HTTPS access to Services.
# Requires an Ingress Controller (nginx, traefik, AWS ALB, etc.)

# Install nginx ingress controller (example)
# kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/...

cat <<'EOF' | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
  namespace: my-app
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: letsencrypt-prod  # auto TLS
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.example.com
      secretName: my-tls-secret
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: my-app-service
                port:
                  number: 80
          - path: /admin
            pathType: Prefix
            backend:
              service:
                name: admin-service
                port:
                  number: 80
EOF

kubectl get ingress
kubectl describe ingress my-app-ingress


# ============================================================
# 8. VOLUMES & PERSISTENT STORAGE
# ============================================================
# Volumes outlive containers within a Pod.
# PersistentVolumes (PV) outlive Pods entirely.

# PersistentVolumeClaim — requests storage from the cluster
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: my-app
spec:
  accessModes:
    - ReadWriteOnce                      # RWO: one node at a time
  storageClassName: standard             # or "gp2", "premium-ssd", etc.
  resources:
    requests:
      storage: 10Gi
EOF

# Access modes:
#   ReadWriteOnce  (RWO) — one node, read-write
#   ReadOnlyMany   (ROX) — many nodes, read-only
#   ReadWriteMany  (RWX) — many nodes, read-write (NFS, EFS)

# Use PVC in a Deployment
cat <<'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: password
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: postgres-pvc
EOF

kubectl get pvc
kubectl get pv
kubectl describe pvc postgres-pvc


# ============================================================
# 9. STATEFULSETS
# ============================================================
# StatefulSets are like Deployments but for stateful apps.
# They give each Pod a stable identity (name, DNS, storage).
# Use for: databases, Kafka, ZooKeeper, Elasticsearch.
#
# Pod names: my-db-0, my-db-1, my-db-2 (stable, ordered)

cat <<'EOF' | kubectl apply -f -
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: my-db
  namespace: my-app
spec:
  serviceName: my-db                     # headless service name
  replicas: 3
  selector:
    matchLabels:
      app: my-db
  template:
    metadata:
      labels:
        app: my-db
    spec:
      containers:
        - name: db
          image: postgres:16
          ports:
            - containerPort: 5432
          volumeMounts:
            - name: data
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:                  # each Pod gets its own PVC
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
---
# Headless Service — stable DNS for each pod
apiVersion: v1
kind: Service
metadata:
  name: my-db
spec:
  clusterIP: None                        # headless — no load balancing
  selector:
    app: my-db
  ports:
    - port: 5432
EOF

# DNS per pod: my-db-0.my-db.my-app.svc.cluster.local
kubectl get statefulset
kubectl describe statefulset my-db


# ============================================================
# 10. JOBS & CRONJOBS
# ============================================================
# Job         — runs a task to completion (batch processing, migrations)
# CronJob     — runs a Job on a schedule (backups, reports)

# Job
cat <<'EOF' | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migrate
  namespace: my-app
spec:
  completions: 1                         # run once
  parallelism: 1                         # max parallel pods
  backoffLimit: 3                        # retries on failure
  ttlSecondsAfterFinished: 300           # auto-cleanup after 5 min
  template:
    spec:
      restartPolicy: OnFailure           # Never or OnFailure (not Always)
      containers:
        - name: migrate
          image: my-app:1.0
          command: ["python", "manage.py", "migrate"]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: url
EOF

# CronJob
cat <<'EOF' | kubectl apply -f -
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-backup
  namespace: my-app
spec:
  schedule: "0 2 * * *"                  # 2am every day (cron syntax)
  concurrencyPolicy: Forbid              # Allow | Forbid | Replace
  successfulJobsHistoryLimit: 3
  failedJobsHistoryLimit: 1
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
            - name: backup
              image: my-backup:1.0
              command: ["/scripts/backup.sh"]
EOF

kubectl get jobs
kubectl get cronjobs
kubectl describe job db-migrate

# Manually trigger a CronJob
kubectl create job --from=cronjob/daily-backup manual-backup-001


# ============================================================
# 11. RESOURCE MANAGEMENT
# ============================================================
# Requests  — guaranteed minimum resources for scheduling
# Limits    — maximum resources a container can use
# LimitRange — default requests/limits for a namespace
# ResourceQuota — total resource cap for a namespace

# LimitRange — enforce defaults and bounds
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: my-app
spec:
  limits:
    - type: Container
      default:
        cpu: "500m"
        memory: "256Mi"
      defaultRequest:
        cpu: "100m"
        memory: "128Mi"
      max:
        cpu: "2"
        memory: "1Gi"
      min:
        cpu: "50m"
        memory: "64Mi"
EOF

# ResourceQuota — total cap for the namespace
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: ResourceQuota
metadata:
  name: namespace-quota
  namespace: my-app
spec:
  hard:
    pods: "20"
    requests.cpu: "4"
    requests.memory: "4Gi"
    limits.cpu: "8"
    limits.memory: "8Gi"
    persistentvolumeclaims: "10"
EOF

# HorizontalPodAutoscaler — scale based on CPU/memory/custom metrics
cat <<'EOF' | kubectl apply -f -
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
  namespace: my-app
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
EOF

kubectl get hpa
kubectl describe hpa my-app-hpa


# ============================================================
# 12. RBAC — ROLE-BASED ACCESS CONTROL
# ============================================================
# Controls who can do what on which resources.
#
# ServiceAccount — identity for a Pod (used in-cluster)
# Role           — permissions within a namespace
# ClusterRole    — permissions cluster-wide
# RoleBinding    — bind Role to a user/group/ServiceAccount
# ClusterRoleBinding — bind ClusterRole cluster-wide

# ServiceAccount
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: my-app-sa
  namespace: my-app
EOF

# Role — namespace-scoped permissions
cat <<'EOF' | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: my-app
rules:
  - apiGroups: [""]                      # "" = core API group
    resources: ["pods", "pods/logs"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list"]
EOF

# RoleBinding — attach Role to a ServiceAccount
cat <<'EOF' | kubectl apply -f -
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: pod-reader-binding
  namespace: my-app
subjects:
  - kind: ServiceAccount
    name: my-app-sa
    namespace: my-app
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
EOF

# Check permissions
kubectl auth can-i list pods --as=system:serviceaccount:my-app:my-app-sa -n my-app
kubectl auth can-i delete pods --as=system:serviceaccount:my-app:my-app-sa -n my-app


# ============================================================
# 13. HEALTH PROBES & POD LIFECYCLE
# ============================================================
# livenessProbe  — restart container if unhealthy
# readinessProbe — remove from Service if not ready
# startupProbe   — protect slow-starting containers

cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: probe-demo
spec:
  containers:
    - name: app
      image: my-app:1.0
      ports:
        - containerPort: 8000

      # Startup probe — checked first; others disabled until it passes
      startupProbe:
        httpGet:
          path: /startup
          port: 8000
        failureThreshold: 30             # allow up to 30 * 10s = 5 min to start
        periodSeconds: 10

      # Readiness probe — remove from Service endpoints if fails
      readinessProbe:
        httpGet:
          path: /ready
          port: 8000
        initialDelaySeconds: 5
        periodSeconds: 10
        failureThreshold: 3

      # Liveness probe — restart container if fails
      livenessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 30
        periodSeconds: 20
        failureThreshold: 3

      # Lifecycle hooks
      lifecycle:
        postStart:
          exec:
            command: ["/bin/sh", "-c", "echo started > /tmp/started"]
        preStop:
          exec:
            command: ["/bin/sh", "-c", "sleep 5"]   # graceful shutdown window

  # Termination grace period (SIGTERM -> wait -> SIGKILL)
  terminationGracePeriodSeconds: 30

  # Probe types available:
  #   httpGet       HTTP GET request
  #   exec          run a command inside the container
  #   tcpSocket     check TCP connection
  #   grpc          gRPC health check
EOF


# ============================================================
# 14. NODE SCHEDULING — LABELS, TAINTS & AFFINITY
# ============================================================
# Control which Pods land on which Nodes.

# --- Labels and nodeSelector (simple) ---
kubectl label node node1 disktype=ssd
kubectl label node node1 zone=us-east-1a

cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: ssd-pod
spec:
  nodeSelector:
    disktype: ssd                        # only schedule on SSD nodes
  containers:
    - name: app
      image: my-app:1.0
EOF

# --- Node Affinity (expressive rules) ---
cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: affinity-pod
spec:
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:   # hard rule
        nodeSelectorTerms:
          - matchExpressions:
              - key: zone
                operator: In
                values: [us-east-1a, us-east-1b]
      preferredDuringSchedulingIgnoredDuringExecution:  # soft rule
        - weight: 80
          preference:
            matchExpressions:
              - key: disktype
                operator: In
                values: [ssd]
    podAntiAffinity:                     # spread pods across nodes
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector:
            matchLabels:
              app: my-app
          topologyKey: kubernetes.io/hostname
  containers:
    - name: app
      image: my-app:1.0
EOF

# --- Taints and Tolerations ---
# Taint a node to repel Pods
kubectl taint nodes node1 gpu=true:NoSchedule
kubectl taint nodes node1 gpu=true:NoSchedule-     # remove taint

cat <<'EOF' | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod
spec:
  tolerations:
    - key: gpu
      operator: Equal
      value: "true"
      effect: NoSchedule               # tolerate the taint
  containers:
    - name: trainer
      image: my-ml-app:1.0
EOF

# Taint effects:
#   NoSchedule       new pods won't be scheduled here
#   PreferNoSchedule try to avoid, not guaranteed
#   NoExecute        evict existing pods + block new ones


# ============================================================
# 15. NETWORK POLICIES
# ============================================================
# NetworkPolicy controls Pod-to-Pod and Pod-to-external traffic.
# Requires a CNI that supports it (Calico, Cilium, Weave).
# Default: all traffic allowed. Once any policy applies, all
# non-matching traffic is denied.

cat <<'EOF' | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: app-network-policy
  namespace: my-app
spec:
  podSelector:
    matchLabels:
      app: my-app                        # applies to these pods

  policyTypes:
    - Ingress
    - Egress

  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend              # only from frontend pods
        - namespaceSelector:
            matchLabels:
              environment: staging       # or from staging namespace
      ports:
        - protocol: TCP
          port: 8000

  egress:
    - to:
        - podSelector:
            matchLabels:
              app: postgres              # only to postgres pods
      ports:
        - protocol: TCP
          port: 5432
    - to:                               # allow DNS lookups
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
EOF

# Deny all ingress to a namespace (default deny)
cat <<'EOF' | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-ingress
  namespace: my-app
spec:
  podSelector: {}                        # applies to all pods
  policyTypes:
    - Ingress
EOF

kubectl get networkpolicies -n my-app


# ============================================================
# 16. USEFUL COMMANDS & HOUSEKEEPING
# ============================================================

# --- Debugging ---
kubectl get events --sort-by=.lastTimestamp -n my-app
kubectl get events --field-selector reason=BackOff
kubectl describe pod my-pod              # status, conditions, events
kubectl top nodes                        # CPU/memory per node (needs metrics-server)
kubectl top pods -n my-app              # CPU/memory per pod
kubectl debug my-pod --image=busybox --copy-to=debug-pod  # debug copy

# --- Diff before applying ---
kubectl diff -f deployment.yaml          # show what would change

# --- Apply vs Create ---
# kubectl apply -f  — declarative, idempotent (recommended)
# kubectl create -f — imperative, fails if resource exists

# --- Dry run ---
kubectl apply -f deployment.yaml --dry-run=client   # validate locally
kubectl apply -f deployment.yaml --dry-run=server   # validate on server

# --- Generate YAML from imperative commands ---
kubectl create deployment my-app --image=my-app:1.0 \
  --dry-run=client -o yaml > deployment.yaml

kubectl expose deployment my-app --port=80 --target-port=8000 \
  --dry-run=client -o yaml > service.yaml

# --- Wait for conditions ---
kubectl wait --for=condition=available deployment/my-app --timeout=60s
kubectl wait --for=condition=ready pod -l app=my-app --timeout=60s
kubectl wait --for=condition=complete job/db-migrate --timeout=120s

# --- Bulk delete ---
kubectl delete all -l app=my-app -n my-app
kubectl delete pods --field-selector status.phase=Failed -n my-app

# --- Patch resources ---
kubectl patch deployment my-app -p '{"spec":{"replicas":4}}'
kubectl patch deployment my-app --type=json \
  -p '[{"op":"replace","path":"/spec/replicas","value":4}]'

# --- Annotate & label ---
kubectl annotate deployment my-app deployment.kubernetes.io/revision=2
kubectl label pod my-pod env=staging --overwrite

# --- Force delete stuck pod ---
kubectl delete pod my-pod --grace-period=0 --force

# --- Switch namespace quickly (kubens from kubectx) ---
# brew install kubectx
# kubens my-app         # switch namespace
# kubectx my-cluster    # switch context


# ============================================================
# QUICK REFERENCE CHEAT SHEET
# ============================================================
#
# KUBECTL
#   kubectl apply -f file.yaml           Create or update
#   kubectl delete -f file.yaml          Delete from file
#   kubectl get <resource> -n <ns>       List resources
#   kubectl describe <resource> <name>   Detailed info + events
#   kubectl logs -f <pod>                Stream logs
#   kubectl exec -it <pod> -- bash       Shell into pod
#   kubectl port-forward <pod> 8080:80   Local port forward
#   kubectl diff -f file.yaml            Preview changes
#   kubectl apply --dry-run=server       Validate without applying
#
# CORE RESOURCES
#   Pod           Smallest unit (1+ containers)
#   Deployment    Manages ReplicaSet + rolling updates
#   Service       Stable network endpoint for pods
#   Ingress       HTTP/S routing from outside cluster
#   ConfigMap     Non-sensitive config data
#   Secret        Sensitive config (base64-encoded)
#   Namespace     Virtual cluster / isolation boundary
#
# WORKLOADS
#   Deployment    Stateless apps, rolling updates
#   StatefulSet   Stateful apps, stable identity + storage
#   DaemonSet     One pod per node (logging, monitoring agents)
#   Job           Run to completion (batch / migration)
#   CronJob       Scheduled Jobs
#
# STORAGE
#   PersistentVolume (PV)         Cluster storage resource
#   PersistentVolumeClaim (PVC)   Request for storage
#   StorageClass                  Dynamic provisioning profile
#   Access modes: RWO / ROX / RWX
#
# SCALING
#   kubectl scale deployment <name> --replicas=N
#   HorizontalPodAutoscaler (HPA)  CPU/memory/custom metrics
#   VerticalPodAutoscaler  (VPA)   Adjust requests/limits
#   KEDA                           Event-driven autoscaling
#
# SCHEDULING
#   nodeSelector                   Simple label matching
#   nodeAffinity                   Expressive label rules
#   podAffinity / podAntiAffinity  Pod-to-pod placement
#   Taints + Tolerations           Repel/attract pods to nodes
#
# HEALTH
#   startupProbe    Protect slow starts
#   readinessProbe  Control Service routing
#   livenessProbe   Restart unhealthy containers
#   Probe types: httpGet / exec / tcpSocket / grpc
#
# RBAC
#   ServiceAccount   Pod identity
#   Role             Namespace-scoped permissions
#   ClusterRole      Cluster-wide permissions
#   RoleBinding / ClusterRoleBinding  Attach to subjects
#
# NETWORK
#   ClusterIP     Internal access only
#   NodePort      External via node IP + static port
#   LoadBalancer  Cloud load balancer
#   Ingress       HTTP routing + TLS termination
#   NetworkPolicy Firewall rules between pods
#
# RESOURCE MANAGEMENT
#   resources.requests   Scheduling guarantee
#   resources.limits     Hard cap
#   LimitRange           Namespace defaults/bounds
#   ResourceQuota        Namespace total cap
