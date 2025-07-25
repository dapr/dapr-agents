def route(path: str, method: str = "GET", **kwargs):
    """
    Decorator to mark an instance method as a FastAPI route.

    Args:
        path (str): The URL path to bind this route to.
        method (str): The HTTP method to use (e.g., 'GET', 'POST'). Defaults to 'GET'.
        **kwargs: Additional arguments passed to FastAPI's `add_api_route`.

    Example:
        @route("/status", method="GET", summary="Show status", tags=["monitoring"])
        def health(self):
            return {"ok": True}
    """

    def decorator(func):
        func._is_fastapi_route = True
        func._route_path = path
        func._route_method = method.upper()
        func._route_kwargs = kwargs
        return func

    return decorator
