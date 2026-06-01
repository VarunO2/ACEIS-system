import requests

API_KEY = "df97b3888fdbfd83c69907fc5b000277"

def get_weather(lat, lon):

    try:

        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}"
            f"&lon={lon}"
            f"&appid={API_KEY}"
            f"&units=metric"
        )

        response = requests.get(url)

        data = response.json()

        print(data)

        if "main" not in data:

            return {
                "temperature": 30,
                "humidity": 50,
                "condition": "Unknown"
            }

        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["main"]
        }

    except Exception as e:

        print(e)

        return {
            "temperature": 30,
            "humidity": 50,
            "condition": "Unknown"
        }