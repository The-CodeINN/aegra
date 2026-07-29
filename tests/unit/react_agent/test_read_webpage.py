"""
tests/unit/react_agent/test_read_webpage.py

Unit tests for the read_webpage tool in graphs/react_agent/tools.py.
All external HTTP calls are mocked — no network required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — add graphs/ root so we can import react_agent directly
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRAPHS_ROOT = PROJECT_ROOT / "graphs"
if str(GRAPHS_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPHS_ROOT))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_httpx_response(status: int = 200, text: str = "", content_type: str = "text/html") -> MagicMock:
    """Build a minimal fake httpx.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.text = text
    resp.url = "https://example.com/page"
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()  # no-op by default
    return resp


def _httpx_status_error(status: int) -> MagicMock:
    """Return an httpx.HTTPStatusError-like mock."""
    import httpx

    request = MagicMock()
    response = MagicMock()
    response.status_code = status
    return httpx.HTTPStatusError("HTTP error", request=request, response=response)


# ---------------------------------------------------------------------------
# Tests: LinkedIn routing
# ---------------------------------------------------------------------------


class TestLinkedInRouting:
    """LinkedIn URLs try Jina Reader first (tier 1); once that's blocked by a
    login wall or fails outright, they fall back to Brave Search (tiers 2/3).
    These tests simulate a login-walled Jina response so the fallback path is
    exercised deterministically — leaving Jina unmocked hit the real network
    and made these tests flaky/order-dependent on whatever r.jina.ai returned
    for a blocked LinkedIn profile at test-run time.
    """

    @staticmethod
    def _login_wall_client() -> MagicMock:
        """A mock httpx.AsyncClient whose Jina Reader response looks like LinkedIn's login wall."""
        login_wall_resp = _make_httpx_response(200, "Sign in to LinkedIn to continue.")
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(return_value=login_wall_resp)
        return mock_client

    @pytest.mark.asyncio
    async def test_linkedin_profile_routes_to_brave_search(self):
        """A /in/<username> URL should fall through to brave_search with the exact URL quoted."""
        from react_agent import tools

        url = "https://www.linkedin.com/in/john-smith/"
        with (
            patch("react_agent.tools.httpx.AsyncClient", return_value=self._login_wall_client()),
            patch.object(
                tools, "brave_search", new=AsyncMock(return_value="John Smith | Data Scientist at Acme")
            ) as mock_brave,
        ):
            result = await tools.read_webpage(url)

        mock_brave.assert_called_once()
        call_arg = mock_brave.call_args[0][0]
        assert call_arg == f'"{url}"'

        assert result["source"] == "brave_search_linkedin"
        assert "John Smith" in result["content"]

    @pytest.mark.asyncio
    async def test_linkedin_post_routes_to_brave_search(self):
        """/posts/<id> URL should also fall through to Brave Search once Jina is blocked."""
        from react_agent import tools

        url = "https://www.linkedin.com/posts/john-smith-abc123_ai-data-science-xyz"
        with (
            patch("react_agent.tools.httpx.AsyncClient", return_value=self._login_wall_client()),
            patch.object(tools, "brave_search", new=AsyncMock(return_value="Post snippet here")) as mock_brave,
        ):
            result = await tools.read_webpage(url)

        mock_brave.assert_called_once()
        call_arg = mock_brave.call_args[0][0]
        assert call_arg == f'"{url}"'
        assert result["source"] == "brave_search_linkedin"

    @pytest.mark.asyncio
    async def test_linkedin_company_routes_to_brave_search(self):
        """/company/<slug> URL should fall through to Brave Search once Jina is blocked."""
        from react_agent import tools

        url = "https://linkedin.com/company/acme-corp"
        with (
            patch("react_agent.tools.httpx.AsyncClient", return_value=self._login_wall_client()),
            patch.object(tools, "brave_search", new=AsyncMock(return_value="Acme Corp on LinkedIn")) as mock_brave,
        ):
            result = await tools.read_webpage(url)

        mock_brave.assert_called_once()
        call_arg = mock_brave.call_args[0][0]
        assert call_arg == f'"{url}"'
        assert result["source"] == "brave_search_linkedin"

    @pytest.mark.asyncio
    async def test_linkedin_fallback_when_brave_fails(self):
        """If Jina is blocked and Brave Search fails entirely, return linkedin_unavailable."""
        from react_agent import tools

        with (
            patch("react_agent.tools.httpx.AsyncClient", return_value=self._login_wall_client()),
            patch.object(tools, "brave_search", new=AsyncMock(side_effect=RuntimeError("api key missing"))),
        ):
            result = await tools.read_webpage("https://linkedin.com/in/someone")

        assert result["error"] == "linkedin_unavailable"
        assert "paste" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_linkedin_fallback_when_brave_returns_empty(self):
        """If Jina is blocked and Brave Search returns 'Search failed', return linkedin_unavailable."""
        from react_agent import tools

        with (
            patch("react_agent.tools.httpx.AsyncClient", return_value=self._login_wall_client()),
            patch.object(tools, "brave_search", new=AsyncMock(return_value="Search failed: 429")),
        ):
            result = await tools.read_webpage("https://linkedin.com/in/someone")

        assert result["error"] == "linkedin_unavailable"


# ---------------------------------------------------------------------------
# Tests: GitHub routing
# ---------------------------------------------------------------------------


class TestGitHubRouting:
    """GitHub profile URLs must hit the REST API, not the HTML page."""

    @pytest.mark.asyncio
    async def test_github_profile_routes_to_rest_api(self):
        """/username URL on github.com should call _fetch_github_profile."""
        from react_agent import tools

        fake_profile = {"source": "github_api", "username": "johndoe", "profile": {}, "repos": []}
        with patch.object(tools, "_fetch_github_profile", new=AsyncMock(return_value=fake_profile)) as mock_gh:
            result = await tools.read_webpage("https://github.com/johndoe")

        mock_gh.assert_called_once_with("johndoe")
        assert result["source"] == "github_api"

    @pytest.mark.asyncio
    async def test_github_api_failure_returns_error(self):
        """If the GitHub API raises, return a github_api_error dict."""
        from react_agent import tools

        with patch.object(tools, "_fetch_github_profile", new=AsyncMock(side_effect=Exception("rate limited"))):
            result = await tools.read_webpage("https://github.com/johndoe")

        assert result["error"] == "github_api_error"
        assert "johndoe" not in result.get("message", "")  # should not leak internal details

    @pytest.mark.asyncio
    async def test_github_bare_url_returns_error(self):
        """github.com with no path should return invalid_github_url."""
        from react_agent import tools

        with patch.object(tools, "_fetch_github_profile", new=AsyncMock()) as mock_gh:
            result = await tools.read_webpage("https://github.com/")

        mock_gh.assert_not_called()
        assert result["error"] == "invalid_github_url"


# ---------------------------------------------------------------------------
# Tests: Jina Reader (general URLs)
# ---------------------------------------------------------------------------


class TestJinaReader:
    """General URLs should use Jina Reader as the primary strategy."""

    @pytest.mark.asyncio
    async def test_general_url_uses_jina_reader(self):
        """A non-LinkedIn/GitHub URL should hit r.jina.ai first."""
        from react_agent import tools

        jina_content = "# My Article\n\n" + "This is the article content from Jina. " * 5  # >100 chars
        jina_resp = _make_httpx_response(200, jina_content, "text/markdown")

        async def fake_get(url, **kwargs):
            if "r.jina.ai" in url:
                return jina_resp
            raise AssertionError(f"Unexpected URL: {url}")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch("react_agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await tools.read_webpage("https://example.com/some-article")

        assert result["source"] == "jina_reader"
        assert "article content" in result["content"]

    @pytest.mark.asyncio
    async def test_jina_failure_falls_back_to_httpx(self):
        """If Jina raises an exception, fall back to direct httpx fetch."""
        from react_agent import tools

        html = "<html><head><title>Hello</title></head><body><p>World</p></body></html>"
        direct_resp = _make_httpx_response(200, html)

        call_count = 0

        async def fake_get(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "r.jina.ai" in url:
                raise ConnectionError("Jina is down")
            return direct_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch("react_agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await tools.read_webpage("https://example.com/page")

        assert result["source"] == "direct_fetch"
        assert call_count == 2  # first Jina, then direct

    @pytest.mark.asyncio
    async def test_jina_short_response_falls_back_to_httpx(self):
        """If Jina returns < 100 chars, treat as empty and fall back."""
        from react_agent import tools

        jina_resp = _make_httpx_response(200, "Too short")  # < 100 chars
        html = "<html><body><p>Real content</p></body></html>"
        direct_resp = _make_httpx_response(200, html)

        async def fake_get(url, **kwargs):
            if "r.jina.ai" in url:
                return jina_resp
            return direct_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch("react_agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await tools.read_webpage("https://example.com/page")

        assert result["source"] == "direct_fetch"

    @pytest.mark.asyncio
    async def test_jina_non_200_falls_back_to_httpx(self):
        """If Jina returns a non-200 status, fall back to direct fetch."""
        from react_agent import tools

        jina_resp = _make_httpx_response(429, "Rate limited")
        html = "<html><body><p>Content</p></body></html>"
        direct_resp = _make_httpx_response(200, html)

        async def fake_get(url, **kwargs):
            return jina_resp if "r.jina.ai" in url else direct_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch("react_agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await tools.read_webpage("https://example.com/page")

        assert result["source"] == "direct_fetch"

    @pytest.mark.asyncio
    async def test_content_truncated_at_8000_chars(self):
        """Content longer than 8000 chars is truncated and truncated=True is set."""
        from react_agent import tools

        long_content = "x" * 10000
        jina_resp = _make_httpx_response(200, long_content, "text/markdown")

        async def fake_get(url, **kwargs):
            if "r.jina.ai" in url:
                return jina_resp
            raise AssertionError("Should not hit httpx fallback")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch("react_agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await tools.read_webpage("https://example.com/long-article")

        assert result["source"] == "jina_reader"
        assert len(result["content"]) == 8000
        assert result["truncated"] is True


# ---------------------------------------------------------------------------
# Tests: Edge cases and validation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_url_returns_error(self):
        from react_agent import tools

        result = await tools.read_webpage("")
        assert result["error"] == "invalid_url"

    @pytest.mark.asyncio
    async def test_non_string_url_returns_error(self):
        from react_agent import tools

        result = await tools.read_webpage(None)  # type: ignore[arg-type]
        assert result["error"] == "invalid_url"

    @pytest.mark.asyncio
    async def test_scheme_is_added_if_missing(self):
        """A URL without a scheme should have https:// prepended."""
        from react_agent import tools

        jina_resp = _make_httpx_response(
            200, "Some content that is definitely longer than one hundred characters for Jina Reader test"
        )

        async def fake_get(url, **kwargs):
            if "r.jina.ai" in url:
                assert "https://example.com" in url
                return jina_resp
            return _make_httpx_response(200, "<html><body>ok</body></html>")

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch("react_agent.tools.httpx.AsyncClient", return_value=mock_client):
            await tools.read_webpage("example.com/page")

    @pytest.mark.asyncio
    async def test_http_error_from_direct_fetch_returns_error_dict(self):
        """A 403/404 from the httpx fallback should return http_error, not raise."""
        import httpx
        from react_agent import tools

        # Jina fails, httpx returns 403
        jina_resp = _make_httpx_response(503, "")
        request_mock = MagicMock()
        response_mock = MagicMock()
        response_mock.status_code = 403

        status_err = httpx.HTTPStatusError("403", request=request_mock, response=response_mock)

        async def fake_get(url, **kwargs):
            if "r.jina.ai" in url:
                return jina_resp
            raise status_err

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get = AsyncMock(side_effect=fake_get)

        with patch("react_agent.tools.httpx.AsyncClient", return_value=mock_client):
            result = await tools.read_webpage("https://example.com/restricted")

        assert result["error"] == "http_error"
        assert result["status_code"] == 403
        assert "automated access" in result["message"].lower()
