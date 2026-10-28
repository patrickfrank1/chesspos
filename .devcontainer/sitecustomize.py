"""Make requests honor REQUESTS_CA_BUNDLE when Session.send is called directly."""

import os

try:
    import requests.adapters
except ImportError:
    pass
else:
    _orig_send = requests.adapters.HTTPAdapter.send

    def _send(self, request, **kwargs):
        bundle = os.environ.get("REQUESTS_CA_BUNDLE")
        if bundle and kwargs.get("verify") is True:
            kwargs["verify"] = bundle
        return _orig_send(self, request, **kwargs)

    requests.adapters.HTTPAdapter.send = _send
