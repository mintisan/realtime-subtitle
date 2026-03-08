def normalize_openai_base_url(base_url):
    """Normalize OpenAI-compatible endpoints to the API root expected by the SDK."""
    if not base_url:
        return None

    normalized = base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/completions", "/responses", "/models"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break

    return normalized or None
