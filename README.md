# daskal_ai
Daskal AI

## Update backend
Since there is only ":latest" tag available in docker hub, to trigger pod image update, change v1 to v2 in deployment.yaml:
``` xml
template:
  metadata:
    labels:
      app: daskal-ai-pod
    annotations:
      rollme: "v1"  # Change this value (e.g., to "v2", "v3", etc.) whenever you need to force a rollout
```

## Add new user
``` bash
kubectl -n argocd edit configmap argocd-cm
```

Add:

``` xml
data:
  accounts.newuser: apiKey, login
```

You can get argocd ip for argocd-server

```bash
kubectl -n argocd get svc
```

Login as admin in argocd server

``` bash
argocd login <ARGOCD_SERVER> --username admin --password <admin-password>
```

Now change newuser password:
```bash
argocd account update-password --account newuser
```

