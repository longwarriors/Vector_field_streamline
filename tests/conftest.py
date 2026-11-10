"""One browser runtime and local server shared across browser test modules."""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Iterator

import pytest
import uvicorn
from playwright.sync_api import Browser, Page, sync_playwright

from browser_support import _browser_presets
from vectorviz.web.app import create_app

# Ports above 1024 that Chromium refuses to load (net::ERR_UNSAFE_PORT).
CHROMIUM_RESTRICTED_PORTS = frozenset(
    {1719, 1720, 1723, 2049, 3659, 4045, 4190, 5060, 5061, 6000, 6566, 6665, 6666, 6667,
     6668, 6669, 6679, 6697, 10080}
)


@pytest.fixture(scope="session")
def frontend_url() -> Iterator[str]:
    while True:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        if listener.getsockname()[1] not in CHROMIUM_RESTRICTED_PORTS:
            break
        listener.close()
    listener.listen(128)
    host, port = listener.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(),
            log_level="error",
            access_log=False,
            lifespan="off",
        )
    )
    thread = threading.Thread(
        target=server.run,
        kwargs={"sockets": [listener]},
        daemon=True,
    )
    thread.start()
    deadline = time.monotonic() + 5.0
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2.0)
        pytest.fail("the browser-test uvicorn server did not start")

    yield f"http://{host}:{port}"

    server.should_exit = True
    thread.join(timeout=5.0)
    assert not thread.is_alive(), "the browser-test uvicorn server did not stop"


@pytest.fixture(scope="session")
def chromium_browser() -> Iterator[Browser]:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def browser_page(chromium_browser: Browser) -> Iterator[tuple[Page, list[str]]]:
    context = chromium_browser.new_context(
        viewport={"width": 1280, "height": 800},
        device_scale_factor=1,
        locale="zh-CN",
    )
    page = context.new_page()
    page_errors: list[str] = []
    page.on("pageerror", lambda error: page_errors.append(str(error)))
    page.route(
        "**/api/presets",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(_browser_presets()),
        ),
    )
    yield page, page_errors
    context.close()
