from geopy.geocoders import Nominatim

def get_location(location_name):

    geolocator = Nominatim(
        user_agent="aceis_system"
    )

    location = geolocator.geocode(
        location_name
    )

    if location:

        return {

            "latitude": location.latitude,

            "longitude": location.longitude,

            "address": location.address

        }

    return None