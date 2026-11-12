from app.api.main import frontend_config


def test_backend_frontend_config_uses_its_own_origin():
    response = frontend_config()

    assert response.media_type == "application/javascript"
    assert b"window.location.origin" in response.body
    assert response.headers["cache-control"] == "no-store"
