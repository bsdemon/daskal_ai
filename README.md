# daskal_ai
Daskal AI

## Update backend
Since there is only ":latest" tag available in docker hub, to trigger pod image update, change v1 to v2:
``` xml
template:
  metadata:
    labels:
      app: daskal-ai-pod
    annotations:
      rollme: "v1"  # Change this value (e.g., to "v2", "v3", etc.) whenever you need to force a rollout
```