# test_weather.py

from geocode import get_location
from weather import get_weather

loc = get_location(
    "Guru Nanak Dev University"
)

print(loc)

weather = get_weather(
    loc["latitude"],
    loc["longitude"]
)

print(weather)