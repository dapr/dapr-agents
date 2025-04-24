from typing import Union
import httpx


class HTTPHelper:
    """
    HTTP operations helper.
    """

    @staticmethod
    def configure_timeout(timeout: Union[int, float, dict]) -> httpx.Timeout:
        """
        Configure the timeout setting for the HTTP client.
        :param timeout: Timeout in seconds or a dictionary of timeout configurations.
        :return: An httpx.Timeout instance configured with the provided timeout.
        """
        if isinstance(timeout, (int, float)):
            return httpx.Timeout(timeout)
        elif isinstance(timeout, dict):
            return httpx.Timeout(**timeout)
        else:
            return httpx.Timeout(30)
