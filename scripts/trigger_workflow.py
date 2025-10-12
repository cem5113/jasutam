import os, requests

def trigger_workflow(repo=None, workflow=None, token=None, ref="main", inputs=None):
    repo = repo or os.getenv("GITHUB_REPO")
    workflow = workflow or os.getenv("GITHUB_WORKFLOW")
    token = token or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    assert repo and workflow and token, "GITHUB_REPO / GITHUB_WORKFLOW / GH_TOKEN zorunlu."

    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    payload = {"ref": ref}
    if inputs:
        payload["inputs"] = inputs

    r = requests.post(url, json=payload, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    if r.status_code != 204:
        raise RuntimeError(f"Trigger failed {r.status_code}: {r.text}")
    return True
