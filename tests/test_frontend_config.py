import os
import subprocess
from pathlib import Path

import pytest

from app.api.main import frontend_config


def test_backend_frontend_config_uses_its_own_origin():
    response = frontend_config()

    assert response.media_type == "application/javascript"
    assert b"window.location.origin" in response.body
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://rag-hybrid-document-qa-abc123.onrender.com",
        "http://localhost:8001",
    ],
)
def test_frontend_build_accepts_render_and_local_api_origins(api_base):
    web_dir = Path(__file__).parents[1] / "web"
    environment = os.environ.copy()
    environment["VITE_API_BASE_URL"] = api_base
    result = subprocess.run(
        ["node", "build.mjs"],
        cwd=web_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    config = (web_dir / "dist" / "config.js").read_text(encoding="utf-8")
    assert api_base in config


def test_frontend_build_rejects_untrusted_api_origin():
    web_dir = Path(__file__).parents[1] / "web"
    environment = os.environ.copy()
    environment["VITE_API_BASE_URL"] = "https://attacker.example"
    result = subprocess.run(
        ["node", "build.mjs"],
        cwd=web_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "must be an onrender.com HTTPS origin" in result.stderr
