# Load testing

`locustfile.py` simulates users who ask a random mix of five example questions (one of them a post-2017 film that must be refused), pausing 2 to 6 seconds between requests, plus an occasional `/health` call. Any `/ask` response other than HTTP 200 counts as a failure.

Install Locust once:

```bash
pip install -e ".[loadtest]"
```

## Two things to set before a test

- **Rate limit.** `/ask` allows 20 requests per minute per client IP (`RATE_LIMIT_PER_MINUTE`). A single load generator is one IP, so it would mostly measure 429s. Raise the limit or set it to `0` for the test, and put it back afterwards.
- **Groq quota.** With `LLM_PROVIDER=groq`, every `/ask` that isn't refused calls Groq. Check your plan's rate limits first; requests over the quota come back as 503 and show up as failures.

## Against the local stack

```bash
# in .env: RATE_LIMIT_PER_MINUTE=0
docker compose up -d --build
python scripts/smoke_test.py http://localhost:8080

locust -f loadtest/locustfile.py --host http://localhost:8080 \
  --headless --users 20 --spawn-rate 2 --run-time 5m \
  --csv results/loadtest/local --only-summary
```

Watch it live on the Grafana dashboard at http://localhost:3000.

## Against Cloud Run

```bash
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$GCP_REGION" --format 'value(status.url)')

# lift the per-IP limit for the duration of the test (it creates a new revision)
gcloud run services update "$SERVICE_NAME" --region "$GCP_REGION" --update-env-vars RATE_LIMIT_PER_MINUTE=0

locust -f loadtest/locustfile.py --host "$SERVICE_URL" \
  --headless --users 20 --spawn-rate 2 --run-time 5m \
  --csv results/loadtest/cloudrun --only-summary

# restore the limit
gcloud run services update "$SERVICE_NAME" --region "$GCP_REGION" --update-env-vars RATE_LIMIT_PER_MINUTE=20
```

The service scales to at most 3 instances, which caps both throughput and cost. Cold starts after a scale-to-zero period appear as slow first requests.

## Summarize

Locust writes `<prefix>_stats.csv`, `<prefix>_stats_history.csv` and `<prefix>_failures.csv`. Turn the stats file into a markdown table:

```bash
python scripts/summarize_loadtest.py results/loadtest/cloudrun_stats.csv --out results/loadtest/cloudrun_summary.md
```

The table has requests, failures, p50, p95 and p99 latency in milliseconds, and requests per second, per endpoint and aggregated. `results/` is gitignored; use `git add -f` to commit a run you want to keep.
