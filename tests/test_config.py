import pytest
from pydantic import ValidationError

from app.config import Settings


def test_cors_requires_exact_http_origins():
    assert Settings(cors_origins="https://example.com,http://localhost:5173").cors_origin_list == [
        "https://example.com",
        "http://localhost:5173",
    ]
    with pytest.raises(ValidationError):
        Settings(cors_origins="*")


def test_trusted_hosts_allow_leading_subdomain_wildcard_only():
    assert "*.onrender.com" in Settings(trusted_hosts="*.onrender.com,localhost").trusted_hosts
    with pytest.raises(ValidationError):
        Settings(trusted_hosts="*;example.com")
