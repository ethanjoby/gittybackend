import base64
import json
import os
import re
from dataclasses import dataclass
from typing import Any

import httpx
from anthropic import AsyncAnthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from playwright.async_api import async_playwright
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="PR Quality Evaluator", version="1.0.0")

GITHUB_API = "https://api.github.com"


class EvaluateRequest(BaseModel):
    issue_url: HttpUrl = Field(..., description="GitHub issue URL")
    pr_url: HttpUrl = Field(..., description="GitHub pull request URL")


class EvaluateResponse(BaseModel):
    score: int
    summary: str
    strengths: list[str]
    issues: list[str]
    actionable_feedback: list[str]
    confidence: str
    model: str
    raw_model_response: dict[str, Any] | None = None


@dataclass
class GitHubRef:
    owner: str
    repo: str
    number: int


ISSUE_URL_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/issues/(\d+)$")
PR_URL_RE = re.compile(r"^https://github\.com/([^/]+)/([^/]+)/pull/(\d+)$")


def parse_issue_url(url: str) -> GitHubRef:
    match = ISSUE_URL_RE.match(url.rstrip("/"))
    if not match:
        raise HTTPException(status_code=400, detail="Invalid GitHub issue URL format")
    owner, repo, number = match.groups()
    return GitHubRef(owner=owner, repo=repo, number=int(number))


def parse_pr_url(url: str) -> GitHubRef:
    match = PR_URL_RE.match(url.rstrip("/"))
    if not match:
        raise HTTPException(status_code=400, detail="Invalid GitHub PR URL format")
    owner, repo, number = match.groups()
    return GitHubRef(owner=owner, repo=repo, number=int(number))


def github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "pr-quality-evaluator",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def fetch_json(client: httpx.AsyncClient, url: str) -> dict[str, Any] | list[Any]:
    resp = await client.get(url)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"GitHub API error: {resp.status_code} {resp.text}")
    return resp.json()


async def fetch_file_content(
    client: httpx.AsyncClient,
    owner: str,
    repo: str,
    path: str,
    ref: str,
) -> str:
    # Use contents API to fetch representative source context for changed files.
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    resp = await client.get(url)
    if resp.status_code >= 400:
        return ""
    data = resp.json()
    if data.get("encoding") != "base64" or "content" not in data:
        return ""
    try:
        decoded = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return decoded[:8000]
    except Exception:
        return ""


async def crawl_with_browserbase(url: str) -> str:
    """
    Crawl rendered page text through Browserbase.

    Required env vars:
      - BROWSERBASE_WS_ENDPOINT: websocket CDP endpoint for Browserbase session
        Example: wss://connect.browserbase.com?apiKey=...&projectId=...
    """
    ws_endpoint = os.getenv("BROWSERBASE_WS_ENDPOINT")
    if not ws_endpoint:
        return ""

    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(ws_endpoint)
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=60000)
            text = await page.locator("main").inner_text(timeout=5000)
            await page.close()
            await browser.close()
            return text[:15000]
    except Exception:
        return ""


async def gather_context(issue: GitHubRef, pr: GitHubRef, issue_url: str, pr_url: str) -> dict[str, Any]:
    if (issue.owner, issue.repo) != (pr.owner, pr.repo):
        raise HTTPException(status_code=400, detail="Issue and PR must belong to the same repository")

    async with httpx.AsyncClient(headers=github_headers(), timeout=30) as client:
        issue_data = await fetch_json(client, f"{GITHUB_API}/repos/{issue.owner}/{issue.repo}/issues/{issue.number}")
        issue_comments = await fetch_json(
            client,
            f"{GITHUB_API}/repos/{issue.owner}/{issue.repo}/issues/{issue.number}/comments?per_page=20",
        )

        pr_data = await fetch_json(client, f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}")
        pr_comments = await fetch_json(
            client,
            f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/issues/{pr.number}/comments?per_page=20",
        )
        pr_review_comments = await fetch_json(
            client,
            f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/comments?per_page=20",
        )
        pr_files = await fetch_json(
            client,
            f"{GITHUB_API}/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/files?per_page=30",
        )

        base_ref = pr_data.get("base", {}).get("sha") or pr_data.get("base", {}).get("ref", "main")
        changed_file_contexts = []
        for f in pr_files[:10]:
            file_path = f.get("filename")
            if not file_path:
                continue
            content = await fetch_file_content(client, pr.owner, pr.repo, file_path, base_ref)
            changed_file_contexts.append(
                {
                    "file": file_path,
                    "status": f.get("status"),
                    "additions": f.get("additions"),
                    "deletions": f.get("deletions"),
                    "patch": (f.get("patch") or "")[:4000],
                    "base_file_excerpt": content,
                }
            )

        readme = await fetch_file_content(client, pr.owner, pr.repo, "README.md", base_ref)

    issue_page_text, pr_page_text = await crawl_with_browserbase(issue_url), await crawl_with_browserbase(pr_url)

    return {
        "repo": f"{pr.owner}/{pr.repo}",
        "issue": {
            "number": issue.number,
            "title": issue_data.get("title"),
            "body": issue_data.get("body"),
            "state": issue_data.get("state"),
            "labels": [l.get("name") for l in issue_data.get("labels", []) if isinstance(l, dict)],
            "comments": [
                {
                    "author": c.get("user", {}).get("login"),
                    "body": c.get("body"),
                }
                for c in issue_comments[:20]
            ],
            "browser_crawl": issue_page_text,
        },
        "pull_request": {
            "number": pr.number,
            "title": pr_data.get("title"),
            "body": pr_data.get("body"),
            "state": pr_data.get("state"),
            "draft": pr_data.get("draft"),
            "mergeable_state": pr_data.get("mergeable_state"),
            "commits": pr_data.get("commits"),
            "additions": pr_data.get("additions"),
            "deletions": pr_data.get("deletions"),
            "changed_files": pr_data.get("changed_files"),
            "issue_comments": [
                {
                    "author": c.get("user", {}).get("login"),
                    "body": c.get("body"),
                }
                for c in pr_comments[:20]
            ],
            "review_comments": [
                {
                    "author": c.get("user", {}).get("login"),
                    "path": c.get("path"),
                    "body": c.get("body"),
                }
                for c in pr_review_comments[:20]
            ],
            "files": changed_file_contexts,
            "browser_crawl": pr_page_text,
        },
        "repository_context": {
            "readme_excerpt": readme,
        },
    }


def build_prompt(context: dict[str, Any]) -> str:
    return f"""
You are a strict senior code reviewer.

Evaluate how well this pull request solves the linked GitHub issue.
Score from 0 to 100.

You must consider:
1) correctness vs issue requirements,
2) code quality and maintainability,
3) edge cases and risk,
4) tests and validation coverage,
5) clarity of implementation and reviewability.

Return ONLY valid JSON with this exact schema:
{{
  "score": <integer 0-100>,
  "summary": "<1-3 sentences>",
  "strengths": ["..."],
  "issues": ["..."],
  "actionable_feedback": ["..."],
  "confidence": "low|medium|high"
}}

Be concrete and technical. Prioritize real risks over style nits.
If context is incomplete, still score based on available evidence and mention uncertainty in issues.

Context:
{json.dumps(context, ensure_ascii=False)}
""".strip()


async def evaluate_with_claude(context: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")

    primary_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    model_candidates = [
        primary_model,
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4-20250514",
    ]
    # Preserve order but dedupe.
    seen = set()
    model_candidates = [m for m in model_candidates if not (m in seen or seen.add(m))]
    client = AsyncAnthropic(api_key=api_key)
    last_exc: Exception | None = None
    message = None
    used_model = None
    for model in model_candidates:
        try:
            message = await client.messages.create(
                model=model,
                max_tokens=1200,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": build_prompt(context),
                    }
                ],
            )
            used_model = model
            break
        except Exception as exc:
            last_exc = exc
            continue

    if message is None or used_model is None:
        tried = ", ".join(model_candidates)
        raise HTTPException(
            status_code=502,
            detail=(
                "Claude API request failed for all candidate models. "
                f"Tried: {tried}. Last error: {last_exc.__class__.__name__}: {str(last_exc)}"
            ),
        ) from last_exc

    text_parts = [blk.text for blk in message.content if getattr(blk, "type", "") == "text"]
    raw_text = "\n".join(text_parts).strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        raw_text = raw_text.replace("json", "", 1).strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail=f"Claude returned non-JSON output: {raw_text[:500]}")

    return {
        "model": used_model,
        "parsed": parsed,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    issue = parse_issue_url(str(req.issue_url))
    pr = parse_pr_url(str(req.pr_url))

    try:
        context = await gather_context(issue, pr, str(req.issue_url), str(req.pr_url))
        model_output = await evaluate_with_claude(context)
        parsed = model_output["parsed"]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected evaluation error: {exc.__class__.__name__}: {str(exc)}",
        ) from exc

    score = int(parsed.get("score", 0))
    score = max(0, min(100, score))

    return EvaluateResponse(
        score=score,
        summary=str(parsed.get("summary", "")),
        strengths=[str(x) for x in parsed.get("strengths", [])],
        issues=[str(x) for x in parsed.get("issues", [])],
        actionable_feedback=[str(x) for x in parsed.get("actionable_feedback", [])],
        confidence=str(parsed.get("confidence", "medium")),
        model=model_output["model"],
        raw_model_response=parsed,
    )


# Run locally:
# uvicorn main:app --reload --port 8000
