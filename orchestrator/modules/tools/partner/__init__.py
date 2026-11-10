"""Partner tools (PRD-275): read-only tools on a partner's own HTTP API, from a pack file.

An operator installs a pack into a workspace and names the credential that holds the
partner's base URL and API key. An agent whose configuration lists the pack is offered its
tools; each call is a GET the server builds, with the key added on the server, so the model
never sees a key or chooses an address.

* ``pack``    the pack format and the tool schemas the model is shown;
* ``request`` the checks on a call's arguments and the URL and headers it becomes;
* ``client``  the GET itself: pinned to a checked address, no redirects, bytes capped;
* ``shape``   the part of the partner's answer the model is given;
* ``answer``  the result a call returns, and how the model reads it;
* ``store``   installed packs, the credential read and the audit row;
* ``offer``   which agents are offered which tools;
* ``runner``  one call from start to finish, on the executor's seam.

Off unless ``PARTNER_TOOLS_ENABLED`` is on. Nothing is imported here: the tool router and
the executor import the modules they wrap with.
"""
