from typing import Optional
from dapr_agents import AgentTool
from datetime import datetime
import requests
import time


class WeatherForecast(AgentTool):
    name: str = "WeatherForecast"
    description: str = "A tool for retrieving the weather/temperature for a given city."

    # Default user agent
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"

    def handle_error(self, response: requests.Response, url: str, stage: str) -> None:
        """Handles error responses and raises a ValueError with detailed information."""
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get data during {stage}. Status: {response.status_code}. "
                f"URL: {url}. Response: {response.text}"
            )
        if not response.json():
            raise ValueError(
                f"No data found during {stage}. URL: {url}. Response: {response.text}"
            )

    def _run(
        self, city: str, state: Optional[str] = None, country: Optional[str] = "usa"
    ) -> dict:
        """
        Retrieves weather data by first fetching geocode data for the city and then fetching weather data.

        Args:
            city (str): The name of the city to get weather for.
            state (Optional[str]): The two-letter state abbreviation (optional).
            country (Optional[str]): The two-letter country abbreviation. Defaults to 'usa'.

        Returns:
            dict: A dictionary containing the city, state, country, and current temperature.
        """
        headers = {"User-Agent": self.user_agent}

        # Construct the geocode URL, conditionally including the state if it's provided
        geocode_url = (
            f"https://nominatim.openstreetmap.org/search?city={city}&country={country}"
        )
        if state:
            geocode_url += f"&state={state}"
        geocode_url += "&limit=1&format=jsonv2"

        # Geocode request
        geocode_response = requests.get(geocode_url, headers=headers)
        self.handle_error(geocode_response, geocode_url, "geocode lookup")

        # Add delay between requests
        time.sleep(2)

        geocode_data = geocode_response.json()
        lat, lon = geocode_data[0]["lat"], geocode_data[0]["lon"]

        # Use different APIs based on the country
        if country.lower() == "usa":
            # Weather.gov request for USA
            weather_gov_url = f"https://api.weather.gov/points/{lat},{lon}"
            weather_response = requests.get(weather_gov_url, headers=headers)
            self.handle_error(weather_response, weather_gov_url, "weather lookup")

            # Add delay between requests
            time.sleep(2)

            weather_data = weather_response.json()
            forecast_url = weather_data["properties"]["forecast"]

            # Forecast request
            forecast_response = requests.get(forecast_url, headers=headers)
            self.handle_error(forecast_response, forecast_url, "forecast lookup")

            forecast_data = forecast_response.json()
            today_forecast = forecast_data["properties"]["periods"][0]

            # Return the weather data along with the city, state, and country
            return {
                "city": city,
                "state": state,
                "country": country,
                "temperature": today_forecast["temperature"],
                "unit": "Fahrenheit",
            }

        else:
            # Met.no API for non-USA countries
            met_no_url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
            weather_response = requests.get(met_no_url, headers=headers)
            self.handle_error(weather_response, met_no_url, "Met.no weather lookup")

            weather_data = weather_response.json()
            temperature_unit = weather_data["properties"]["meta"]["units"][
                "air_temperature"
            ]
            today_forecast = weather_data["properties"]["timeseries"][0]["data"][
                "instant"
            ]["details"]["air_temperature"]

            # Return the weather data along with the city, state, and country
            return {
                "city": city,
                "state": state,
                "country": country,
                "temperature": today_forecast,
                "unit": temperature_unit,
            }


class HistoricalWeather(AgentTool):
    name: str = "HistoricalWeather"
    description: str = (
        "A tool for retrieving historical weather data (temperature) for a given city."
    )

    # Default user agent
    user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15"

    def handle_error(self, response: requests.Response, url: str, stage: str) -> None:
        """Handles error responses and raises a ValueError with detailed information."""
        if response.status_code != 200:
            raise ValueError(
                f"Failed to get data during {stage}. Status: {response.status_code}. "
                f"URL: {url}. Response: {response.text}"
            )
        if not response.json():
            raise ValueError(
                f"No data found during {stage}. URL: {url}. Response: {response.text}"
            )

    def _run(
        self,
        city: str,
        state: Optional[str] = None,
        country: Optional[str] = "usa",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict:
        """
        Retrieves historical weather data for the city by first fetching geocode data and then historical weather data.

        Args:
            city (str): The name of the city to get weather for.
            state (Optional[str]): The two-letter state abbreviation (optional).
            country (Optional[str]): The two-letter country abbreviation. Defaults to 'usa'.
            start_date (Optional[str]): Start date for historical data (YYYY-MM-DD format).
            end_date (Optional[str]): End date for historical data (YYYY-MM-DD format).

        Returns:
            dict: A dictionary containing the city, state, country, and historical temperature data.
        """
        headers = {"User-Agent": self.user_agent}

        # Validate dates
        current_date = datetime.now().strftime("%Y-%m-%d")
        if start_date >= current_date or end_date >= current_date:
            raise ValueError(
                "Both start_date and end_date must be earlier than the current date."
            )

        if (
            datetime.strptime(end_date, "%Y-%m-%d")
            - datetime.strptime(start_date, "%Y-%m-%d")
        ).days > 30:
            raise ValueError(
                "The time span between start_date and end_date cannot exceed 30 days."
            )

        # Construct the geocode URL, conditionally including the state if it's provided
        geocode_url = (
            f"https://nominatim.openstreetmap.org/search?city={city}&country={country}"
        )
        if state:
            geocode_url += f"&state={state}"
        geocode_url += "&limit=1&format=jsonv2"

        # Geocode request
        geocode_response = requests.get(geocode_url, headers=headers)
        self.handle_error(geocode_response, geocode_url, "geocode lookup")

        # Add delay between requests
        time.sleep(2)

        geocode_data = geocode_response.json()
        lat, lon = geocode_data[0]["lat"], geocode_data[0]["lon"]

        # Historical weather request
        historical_weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m"
        weather_response = requests.get(historical_weather_url, headers=headers)
        self.handle_error(
            weather_response, historical_weather_url, "historical weather lookup"
        )

        weather_data = weather_response.json()

        # Extract time and temperature data
        timestamps = weather_data["hourly"]["time"]
        temperatures = weather_data["hourly"]["temperature_2m"]
        temperature_unit = weather_data["hourly_units"]["temperature_2m"]

        # Combine timestamps and temperatures into a dictionary
        temperature_data = {
            timestamps[i]: temperatures[i] for i in range(len(timestamps))
        }

        # Return the structured weather data along with the city, state, country
        return {
            "city": city,
            "state": state,
            "country": country,
            "start_date": start_date,
            "end_date": end_date,
            "temperature_data": temperature_data,
            "unit": temperature_unit,
        }
