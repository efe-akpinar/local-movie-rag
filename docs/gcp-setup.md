# One-time Google Cloud setup

Run these once, in order, from a shell with the [gcloud CLI](https://cloud.google.com/sdk/docs/install) and the [GitHub CLI](https://cli.github.com/) logged in (`gcloud auth login`, `gh auth login`). Replace every `<placeholder>`.

After this, every push to `main` runs the quality gate, builds the image and deploys to Cloud Run. GitHub never holds a Google key: Actions authenticates through Workload Identity Federation, and the Groq key lives only in Secret Manager.

## 0. Variables

```bash
export PROJECT_ID="<movie-rag-unique-id>"          # globally unique, 6-30 chars
export BILLING_ACCOUNT_ID="<XXXXXX-XXXXXX-XXXXXX>"  # gcloud billing accounts list
export REGION="europe-west1"
export AR_REPO="movie-rag"
export SERVICE_NAME="movie-rag"
export INDEX_BUCKET="${PROJECT_ID}-index"
export GITHUB_REPO="efe-akpinar/local-movie-rag"
export DEPLOYER_SA="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
export RUNTIME_SA="movie-rag-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
```

## 1. Create the project and link billing

```bash
gcloud projects create "$PROJECT_ID" --name="Movie RAG"
gcloud billing projects link "$PROJECT_ID" --billing-account="$BILLING_ACCOUNT_ID"
gcloud config set project "$PROJECT_ID"
export PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
```

## 2. Budget alert

A budget sends email alerts to the billing admins; it does not stop spending. The currency must match your billing account's currency.

```bash
gcloud services enable billingbudgets.googleapis.com
gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT_ID" \
  --display-name="movie-rag monthly budget" \
  --budget-amount=<10USD> \
  --filter-projects="projects/${PROJECT_ID}" \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0
```

## 3. Enable the APIs

`iam` and `sts` are needed in addition to the five services for creating service accounts and for the Workload Identity Federation token exchange.

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  iamcredentials.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  sts.googleapis.com
```

## 4. Artifact Registry repository and private index bucket

```bash
gcloud artifacts repositories create "$AR_REPO" \
  --repository-format=docker --location="$REGION" --description="movie-rag images"

gcloud storage buckets create "gs://${INDEX_BUCKET}" \
  --location="$REGION" --uniform-bucket-level-access --public-access-prevention
```

## 5. Groq API key in Secret Manager

Create a key at https://console.groq.com/keys. `read -s` keeps it off the screen and out of your shell history.

```bash
read -rs GROQ_API_KEY   # paste the key, press Enter
printf '%s' "$GROQ_API_KEY" | gcloud secrets create groq-api-key \
  --replication-policy=automatic --data-file=-
unset GROQ_API_KEY
```

To rotate it later: `printf '%s' "$NEW_KEY" | gcloud secrets versions add groq-api-key --data-file=-`, then redeploy. The service reads `:latest` when a new revision starts.

## 6. Service accounts (least privilege)

```bash
gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer"
gcloud iam service-accounts create movie-rag-runtime --display-name="movie-rag Cloud Run runtime"
```

**Runtime account** (what the running service is): it can read the Groq secret and nothing else.

```bash
gcloud secrets add-iam-policy-binding groq-api-key \
  --member="serviceAccount:${RUNTIME_SA}" --role=roles/secretmanager.secretAccessor
```

**Deployer account** (what GitHub Actions acts as):

```bash
# Deploy Cloud Run services and make them public (--allow-unauthenticated sets the service IAM policy,
# which roles/run.developer cannot do).
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${DEPLOYER_SA}" --role=roles/run.admin

# Push images to this one repository.
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" --location="$REGION" \
  --member="serviceAccount:${DEPLOYER_SA}" --role=roles/artifactregistry.writer

# Read the index artifacts from this one bucket.
gcloud storage buckets add-iam-policy-binding "gs://${INDEX_BUCKET}" \
  --member="serviceAccount:${DEPLOYER_SA}" --role=roles/storage.objectViewer

# Deploy revisions that run as the runtime account.
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" \
  --member="serviceAccount:${DEPLOYER_SA}" --role=roles/iam.serviceAccountUser
```

## 7. Workload Identity Federation for GitHub Actions

The provider only accepts tokens from workflows in `efe-akpinar/local-movie-rag` running on `main`.

```bash
gcloud iam workload-identity-pools create github \
  --location=global --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc local-movie-rag \
  --location=global \
  --workload-identity-pool=github \
  --display-name="local-movie-rag" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository == '${GITHUB_REPO}' && assertion.ref == 'refs/heads/main'"

gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/${GITHUB_REPO}"

export WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/providers/local-movie-rag"
```

## 8. GitHub repository variables

These are variables, not secrets: none of them is sensitive on its own.

```bash
gh variable set GCP_PROJECT_ID --repo "$GITHUB_REPO" --body "$PROJECT_ID"
gh variable set GCP_REGION     --repo "$GITHUB_REPO" --body "$REGION"
gh variable set WIF_PROVIDER   --repo "$GITHUB_REPO" --body "$WIF_PROVIDER"
gh variable set DEPLOYER_SA    --repo "$GITHUB_REPO" --body "$DEPLOYER_SA"
gh variable set RUNTIME_SA     --repo "$GITHUB_REPO" --body "$RUNTIME_SA"
gh variable set AR_REPO        --repo "$GITHUB_REPO" --body "$AR_REPO"
gh variable set INDEX_BUCKET   --repo "$GITHUB_REPO" --body "$INDEX_BUCKET"
gh variable set SERVICE_NAME   --repo "$GITHUB_REPO" --body "$SERVICE_NAME"
```

The deploy workflow is skipped until `GCP_PROJECT_ID` is set, and fails with a clear message if any other variable is missing.

`GROQ_MODEL` is optional. Without it the service uses `openai/gpt-oss-120b`. When Groq retires a model (requests then fail with HTTP 404 and `/ask` returns 503), pick a current one from [console.groq.com/docs/models](https://console.groq.com/docs/models), set it and redeploy. No code change is needed:

```bash
gh variable set GROQ_MODEL --repo "$GITHUB_REPO" --body "openai/gpt-oss-120b"
gh workflow run deploy.yml --repo "$GITHUB_REPO" --ref main
```

## 9. Upload the index

The deploy workflow downloads `gs://$INDEX_BUCKET/index/{index.faiss,records.jsonl,manifest.json}`. If you already built the index locally into `artifacts/index/`, upload those files without rebuilding:

```bash
pip install -e ".[gcs]"
gcloud auth application-default login
python scripts/build_index.py --upload-only --gcs-bucket "$INDEX_BUCKET" --gcs-project "$PROJECT_ID"
gcloud storage ls -l "gs://${INDEX_BUCKET}/index/"
```

`--upload-only` first reloads the three files and stops with an error, uploading nothing, unless they agree with each other: vector count, record count and manifest `doc_count` all equal 41,328, the embedding model matches, and both files match the sha256 hashes in `manifest.json`.

Without a local build, build and upload on a Colab GPU runtime instead, with `movies_metadata.csv` and `credits.csv` uploaded to `/content/data`:

```python
from google.colab import auth
auth.authenticate_user()

!git clone https://github.com/efe-akpinar/local-movie-rag.git
%cd local-movie-rag
!pip install -q -e ".[gcs]"
!python scripts/build_index.py /content/data --gcs-bucket <INDEX_BUCKET> --gcs-project <PROJECT_ID>
```

The script stops with an error unless the corpus has exactly 41,328 documents. To use the same artifacts locally (for `docker compose`):

```bash
mkdir -p artifacts/index
gcloud storage cp "gs://${INDEX_BUCKET}/index/*" artifacts/index/
```

## 10. First deploy

Merge the pull request into `main`, or start it by hand:

```bash
gh workflow run deploy.yml --repo "$GITHUB_REPO" --ref main
gh run watch --repo "$GITHUB_REPO"
URL="$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')"
echo "$URL"
```

Then check that the rate limiter keys on your address and not on what a caller puts in `X-Forwarded-For` (see `client_ip` in `movie_rag/api.py`). Each request below forges a different first entry. Empty questions are rejected after the rate-limit check, so this makes no Groq calls:

```bash
for i in $(seq 1 21); do
  curl -s -o /dev/null -w '%{http_code} ' -X POST "$URL/ask" -H 'Content-Type: application/json' \
    -H "X-Forwarded-For: 10.0.0.$i" -d '{"question": ""}'
done; echo
```

Expect twenty `422` and then a `429`. Each instance keeps its own counter, so if the requests were spread over two instances you may see only `422`; run it again once the service is warm. Twenty-one `422` on a single warm instance would mean the forged entries are being used as the key.

## Cost notes

- Cloud Run scales to zero (`--min-instances=0`) and is capped at 3 instances, so idle time costs nothing and a traffic spike cannot run away.
- Artifact Registry stores image layers; layers shared between builds are stored once, so repeated deploys mainly add the small application layer. Delete old images with `gcloud artifacts docker images delete` if storage grows.
- The index bucket holds well under 1 GB.
