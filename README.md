# PR Quality Evaluator

FastAPI service that evaluates how well a GitHub pull request addresses a linked GitHub issue.

It gathers issue/PR context from the GitHub API, optionally crawls rendered pages through Browserbase, and asks Claude to produce a structured quality assessment.

## Features

- Validates GitHub issue and PR URLs
- Fetches issue details, PR metadata, comments, review comments, and changed files
- Pulls base-file excerpts for changed files
- Optionally crawls issue/PR pages via Browserbase (Playwright + CDP)
- Returns a strict JSON evaluation with score, strengths, issues, and actionable feedback

## Requirements

- Python 3.10+
- A GitHub token (`GITHUB_TOKEN`) recommended to avoid API limits
- Anthropic API key (`ANTHROPIC_API_KEY`)
- Optional Browserbase CDP websocket endpoint (`BROWSERBASE_WS_ENDPOINT`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install python-dotenv
```

Playwright browsers are also required:

```bash
playwright install chromium
```

## Environment Variables

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_key
GITHUB_TOKEN=your_github_token
BROWSERBASE_WS_ENDPOINT=wss://connect.browserbase.com?apiKey=...&projectId=...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

Notes:
- `ANTHROPIC_API_KEY` is required.
- `GITHUB_TOKEN` is optional but strongly recommended.
- `BROWSERBASE_WS_ENDPOINT` is optional. If omitted, browser crawl fields are empty.
- `ANTHROPIC_MODEL` is optional and defaults to `claude-3-5-sonnet-20241022`.

## Run

```bash
uvicorn main:app --reload --port 8000
```

## API

### Health check

`GET /health`

Response:

```json
{ "status": "ok" }
```

### Evaluate PR quality

`POST /evaluate`

Request body:

```json
{
  "issue_url": "https://github.com/owner/repo/issues/123",
  "pr_url": "https://github.com/owner/repo/pull/456"
}
```

Response shape:

```json
{
  "score": 84,
  "summary": "Short technical summary.",
  "strengths": ["..."],
  "issues": ["..."],
  "actionable_feedback": ["..."],
  "confidence": "medium",
  "model": "claude-3-5-sonnet-20241022",
  "raw_model_response": {}
}
```

## Example cURL

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "issue_url": "https://github.com/owner/repo/issues/123",
    "pr_url": "https://github.com/owner/repo/pull/456"
  }'
```
