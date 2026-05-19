import os
from pathlib import Path


def load_local_env() -> None:
    """Load KEY=VALUE pairs from local env files if shell env is unset.

    This avoids hardcoding API keys in Python source files while removing the
    need to export them manually for every terminal session.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here / ".env.local",
        here.parent / ".env.local",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def get_openai_compatible_kwargs() -> dict[str, str | None]:
    """Return API kwargs for OpenAI-compatible providers.

    Supported env vars:
    - OPENAI_API_KEY + OPENAI_BASE_URL / OPENAI_API_BASE
    - NVIDIA_NIM_API_KEY / NVIDIA_API_KEY
    - GEMINI_API_KEY
    - VERTEX_PROJECT_ID + Google ADC or VERTEX_ACCESS_TOKEN
    - GROQ_API_KEY
    - OPENROUTER_API_KEY
    """
    load_local_env()

    vertex_project = os.environ.get("VERTEX_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if vertex_project:
        location = os.environ.get("VERTEX_LOCATION", "global")
        token = os.environ.get("VERTEX_ACCESS_TOKEN") or get_vertex_access_token()
        return {
            "api_key": token,
            "base_url": (
                "https://aiplatform.googleapis.com/v1/"
                f"projects/{vertex_project}/locations/{location}/endpoints/openapi"
            ),
        }

    api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("NVIDIA_NIM_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    base_url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("NVIDIA_NIM_BASE_URL")
        or os.environ.get("NVIDIA_BASE_URL")
    )
    if base_url is None and (
        os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    ):
        base_url = "https://integrate.api.nvidia.com/v1"
    elif base_url is None and os.environ.get("GEMINI_API_KEY"):
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif base_url is None and os.environ.get("GROQ_API_KEY"):
        base_url = "https://api.groq.com/openai/v1"
    elif base_url is None and os.environ.get("OPENROUTER_API_KEY"):
        base_url = "https://openrouter.ai/api/v1"

    return {"api_key": api_key, "base_url": base_url}


def get_vertex_access_token() -> str:
    """Get a Vertex AI access token from Google Application Default Credentials."""
    try:
        from google.auth import default
        import google.auth.transport.requests
    except ImportError as exc:
        raise RuntimeError(
            "Vertex AI auth requires the `google-auth` package. Install it with "
            "`python3 -m pip install google-auth` and run "
            "`gcloud auth application-default login`."
        ) from exc

    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    credentials.refresh(google.auth.transport.requests.Request())
    return credentials.token
