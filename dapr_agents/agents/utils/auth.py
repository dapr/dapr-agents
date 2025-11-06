import requests
import os


def construct_auth_headers(auth_url, grant_type="client_credentials", **kwargs):
    """
    Construct authorization headers for API requests.

    :param auth_url: The authorization URL.
    :param grant_type: The type of OAuth grant (default is 'client_credentials').
    :param kwargs: Additional parameters for the POST request body.

    :return: A dictionary containing the Authorization header.
    """

    # Define default parameters based on the grant_type
    data = {
        "grant_type": grant_type,
    }

    # Defaults for client_credentials grant type
    if grant_type == "client_credentials":
        data.update(
            {
                "client_id": kwargs.get("client_id", os.getenv("CLIENT_ID")),
                "client_secret": kwargs.get(
                    "client_secret", os.getenv("CLIENT_SECRET")
                ),
            }
        )

    # Add any additional data passed in kwargs
    data.update(kwargs)

    # POST request to obtain the access token
    auth_response = requests.post(auth_url, data=data)

    # Check if the response was successful
    auth_response.raise_for_status()

    # Convert the response to JSON
    auth_response_data = auth_response.json()

    # Extract the access token
    access_token = auth_response_data.get("access_token")

    if not access_token:
        raise ValueError("No access token found in the response")

    return {"Authorization": f"Bearer {access_token}"}
