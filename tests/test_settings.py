"""Web settings read list values from .env as comma-separated text, as .env.example writes them."""

from envforge.web.settings import WebSettings


def test_list_settings_are_comma_separated(monkeypatch):
    monkeypatch.setenv("BASE_URL_ALLOW_CIDRS", "100.64.0.0/10")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:3000, https://envforge.example.com")
    s = WebSettings()
    assert s.BASE_URL_ALLOW_CIDRS == ["100.64.0.0/10"]
    assert s.CORS_ORIGINS == ["http://localhost:3000", "https://envforge.example.com"]
    monkeypatch.setenv("BASE_URL_ALLOW_CIDRS", "")
    assert WebSettings().BASE_URL_ALLOW_CIDRS == []
