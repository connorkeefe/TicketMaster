import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logger import logger

from camoufox.sync_api import Camoufox

from playwright.sync_api import Response

import time

from urllib.parse import urlparse

def parse_proxy(proxy_str: str) -> dict:
    """
    Parse a proxy string of the form:
      'http://username:password@server:port'
    Returns a dict with username, password, server, port.
    """
    if not proxy_str:
        return {}

    # Ensure it's only the URL part (not the dict syntax)
    proxy_str = proxy_str.strip().strip("'\"")  # remove quotes
    if proxy_str.startswith("{"):
        # Handle string like "'http': f'http://user:pass@host:port'"
        proxy_str = proxy_str.split(":", 1)[-1].strip()
        proxy_str = proxy_str.strip("f").strip("'\"")

    parsed = urlparse(proxy_str)

    return {
        "scheme": parsed.scheme,
        "username": parsed.username,
        "password": parsed.password,
        "server": parsed.hostname,
        "port": parsed.port,
    }



class StealthWeb:
    def __init__(self, base_url=None, type_call='inventory', proxy=None):
        self.headers = {}
        self.base_url = base_url
        self.proxy = proxy
        if type_call == "inventory":
            self.request_id = "inventorytypes%20offertypes%"
        else:
            self.request_id = "inventorytypes%20offer&q=available&show=listpricerange"

        self.camo_kwargs = {"humanize": True}
        if getattr(self, "proxy", None):
            proxy_details = parse_proxy(self.proxy)
            # Pass proxy and geoip when a proxy is configured on the instance
            self.camo_kwargs["proxy"] = {
                "server": f"{proxy_details['scheme']}://{proxy_details['server']}:{proxy_details['port']}",  # e.g. "http://user:pass@host:port"
                # If your Camoufox expects username/password as separate keys, set them here:
                "username": proxy_details["username"],
                "password": proxy_details["password"],
            }
            # Some versions accept geoip as top-level kwarg; include both places in case:
            self.camo_kwargs["geoip"] = True

    def _matches(self, url: str, match_substr: str | None = None) -> bool:
        """
        Decide if a URL is the one we care about.
        If match_substr is provided, use that; else fall back to self.request_id.
        """
        needle = match_substr if match_substr is not None else self.request_id
        return needle in url

    def _handle_request(self, request):
        # logger.info(request.url)
        if self.request_id in request.url:
            # logger.info("Here")
            # logger.info(request.headers.items())
            self.headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in ["content-length", "transfer-encoding", "set-cookie"]
            }

    def get_headers(self):
        """
        Open a page in a stealth Camoufox browser and extract cookies.

        Args:
            url (str): The URL to open.
            output_file (str): Path to save cookies as JSON.
        """
        try:
            # Launch Camoufox with full stealth mode
            with Camoufox(**self.camo_kwargs) as browser:
                # Open new tab / page
                page = browser.new_page()
                self.apply_stealth(page)
                # page.goto("https://google.com")
                page.on("request", self._handle_request)
                page.goto(self.base_url)

                time.sleep(2)

                # Optional: wait until all network requests are done
                page.wait_for_load_state("networkidle")

                time.sleep(1)

                # You can also wait for a specific element to appear if needed
                # page.wait_for_selector("body")
                # Pause a moment (some sites add cookies with JS after load)
                # Retrieve cookies from the current browser context
        except Exception as e:
            logger.error(f"Failed to get headers: {e}")

    def apply_stealth(self, page):
        # navigator.webdriver -> False
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => false });
        """)

        # languages
        page.add_init_script("""
            Object.defineProperty(navigator, 'language', { get: () => 'en-US' });
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        """)

        # plugins
        page.add_init_script("""
            Object.defineProperty(navigator, 'plugins', {
              get: () => [1,2,3,4,5].map(i => ({ name: 'Plugin'+i }))
            });
        """)

        # permissions (notifications) to avoid “denied” default in headless
        page.add_init_script("""
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
              parameters.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : originalQuery(parameters)
            );
        """)

        # WebGL vendor/renderer
        page.add_init_script("""
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(param){
              if (param === 37445) return 'Intel Inc.';        // UNMASKED_VENDOR_WEBGL
              if (param === 37446) return 'Intel Iris OpenGL'; // UNMASKED_RENDERER_WEBGL
              return getParameter.call(this, param);
            };
        """)

        # chrome runtime (for UA parity)
        page.add_init_script("""
            if (!window.chrome) window.chrome = { runtime: {} };
        """)

    def get_response_both(
            self,
            page_url: str,
            primary_substr: str | None = None,
            secondary_substr: str | None = None,
            timeout_ms: int = 15000,
            wait_secondary: bool | None = None,  # overrides env if provided
    ):
        """
        Wait for one or two responses in a single navigation.

        Returns:
          - if waiting for secondary (toggle on or wait_secondary=True):
              ((primary_response, primary_body), (secondary_response, secondary_body))
          - if not waiting (toggle off or wait_secondary=False):
              (primary_response, primary_body)

        Bodies are generated IDENTICALLY to your get_response: response.json().
        If JSON parsing fails, we return (None, None) for that tuple (to keep parity with your try/except).
        """

        # decide whether we want the secondary based on env unless explicitly overridden
        if wait_secondary is None:
            wait_secondary = False

        primary_substr = primary_substr or self.request_id
        want_secondary = bool(wait_secondary and secondary_substr)

        # storage
        primary_resp, primary_body = None, None
        secondary_resp, secondary_body = None, None

        # simple matchers that mirror your _matches logic
        def _is_primary(url: str) -> bool:
            try:
                return self._matches(url, primary_substr)
            except Exception:
                return False

        def _is_secondary(url: str) -> bool:
            if not want_secondary:
                return False
            try:
                return self._matches(url, secondary_substr)
            except Exception:
                return False

        with Camoufox(**self.camo_kwargs) as browser:
            page = browser.new_page()
            self.apply_stealth(page)

            # capture outgoing request headers just like you do now
            page.on("request", self._handle_request)

            # response listeners that grab the FIRST match of each kind
            def _on_response(resp: Response):
                nonlocal primary_resp, primary_body, secondary_resp, secondary_body
                try:
                    url = resp.url
                    if primary_resp is None and _is_primary(url):
                        try:
                            primary_resp = resp
                            primary_body = resp.json()  # IDENTICAL generation to your current code
                        except Exception:
                            # mirror your error path: return nothing for this one
                            primary_resp, primary_body = None, None
                    elif want_secondary and secondary_resp is None and _is_secondary(url):
                        try:
                            secondary_resp = resp
                            secondary_body = resp.json()  # IDENTICAL generation
                        except Exception:
                            secondary_resp, secondary_body = None, None
                except Exception:
                    # ignore; keep searching until timeout
                    pass

            page.on("response", _on_response)

            # navigate AFTER listeners are armed
            start = time.time()
            deadline = start + (timeout_ms / 1000.0)
            try:
                page.goto(page_url, timeout=timeout_ms)
            except Exception as e:
                logger.error(f"Navigation error: {e}")

            # wait loop until we have what we need or timeout
            def _done() -> bool:
                got_primary = primary_resp is not None
                if want_secondary:
                    return got_primary and (secondary_resp is not None)
                return got_primary

            while time.time() < deadline and not _done():
                page.wait_for_timeout(100)

        # return shape:
        return (primary_resp, primary_body), (secondary_resp, secondary_body)

    def get_response(self, page_url: str, timeout_ms: int = 15000):
        """
        Visit `page_url`, wait for the network call whose URL contains `match_substr`
        (or self.request_id if not provided), and return both request headers and response.

        Returns:
            {
                "status": int,
                "url": str,
                "headers_sent": dict,
                "headers_received": dict,
                "body": Any  # JSON if possible, else text, else bytes
            }
        """
        self.headers = {}  # reset per call

        with Camoufox(**self.camo_kwargs) as browser:
            page = browser.new_page()
            self.apply_stealth(page)

            # Listen for the outgoing request so we can capture the headers
            page.on("request", self._handle_request)

            # Expect the specific response that matches our predicate.
            # IMPORTANT: set up the expectation BEFORE the action that triggers it.
            def _predicate(resp):
                try:
                    return self._matches(resp.url, self.request_id)
                except Exception:
                    return False
            try:
                with page.expect_response(_predicate, timeout=timeout_ms) as resp_info:
                    page.goto(page_url, timeout=timeout_ms)

                    response = resp_info.value
                    body = response.json()

            except Exception as e:
                logger.error(f"Failed to get response for {self.request_id} with error: {e}")
                response = None
                body = None

            return response, body
