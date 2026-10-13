"""Wildlife Insights URL resolution using centralized credentials."""

import re

import requests

from src.utils.credentials import get_wildlife_insights_token


def _resolve_wildlife_insights_url(
    url, cookie=None, token=None, photo_id=None, token_getter=None
):
    """Resolves a Wildlife Insights public web viewer URL into a signed Google Cloud Storage image URL."""
    if token_getter is None:
        token_getter = get_wildlife_insights_token

    if not url or "wildlifeinsights.org" not in url:
        return None

    m_dl = re.search(r"/download/(\d+)", url)
    m_proj = re.search(r"/project/(\d+)", url)
    m_img = re.search(r"/data-files/([a-zA-Z0-9\-]+)", url)

    dl_id = int(m_dl.group(1)) if m_dl else None
    proj_id = int(m_proj.group(1)) if m_proj else None
    img_uuid = m_img.group(1) if m_img else str(photo_id or "")

    if not dl_id or not img_uuid:
        return None

    auth_token = token or token_getter(cookie)
    if not auth_token:
        return None

    gql_ep = "https://api.wildlifeinsights.org/graphql"
    query = """query getDataFilePublicDownloadUrl($downloadId: Int!, $projectId: Int, $imageUUID: String!) {
  getDataFilePublicDownloadUrl(downloadId: $downloadId, projectId: $projectId, imageUUID: $imageUUID) {
    url
  }
}"""
    variables = {
        "downloadId": dl_id,
        "projectId": proj_id,
        "imageUUID": img_uuid,
    }
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0",
    }

    try:
        r = requests.post(
            gql_ep,
            json={"query": query, "variables": variables},
            headers=headers,
            timeout=10,
        )
        if r.status_code == 401:
            auth_token = token_getter(cookie, refresh=True)
            if auth_token:
                headers["Authorization"] = f"Bearer {auth_token}"
                r = requests.post(
                    gql_ep,
                    json={"query": query, "variables": variables},
                    headers=headers,
                    timeout=10,
                )
        if r.status_code == 200:
            res_data = r.json()
            if (
                "data" in res_data
                and res_data["data"]
                and res_data["data"].get("getDataFilePublicDownloadUrl")
            ):
                return res_data["data"]["getDataFilePublicDownloadUrl"].get("url")
    except Exception:
        pass
    return None


def resolve_wildlife_insights_url(url, cookie=None, token=None, photo_id=None):
    """Resolve a Wildlife Insights viewer URL to its signed image URL."""
    return _resolve_wildlife_insights_url(
        url,
        cookie=cookie,
        token=token,
        photo_id=photo_id,
        token_getter=get_wildlife_insights_token,
    )
