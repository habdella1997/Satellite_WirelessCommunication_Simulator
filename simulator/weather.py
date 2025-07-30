import requests
import constants as C

def get_weather_stats(lat, lon):
    payload = {
        'lat': str(lat),
        'lon': str(lon),
        'appid': C.WEATHER_API_KEY,
        'units': 'metric'
    }

    try:
        r = requests.get('https://api.openweathermap.org/data/2.5/weather', params=payload)
        print("Requesting URL:", r.url)
        print("Response status:", r.status_code)

        if r.status_code != 200:
            raise Exception(f"API request failed: {r.status_code}, {r.text}")

        data = r.json()
        temp     = float(data['main']['temp'])
        humidity = float(data['main']['humidity'])
        pressure = float(data['main']['pressure'])

    except Exception as e:
        print(f"Weather API error: {e}")
        print("Using default average values.")

        # ✅ Fallback to average values
        temp     = 15.0       # °C
        humidity = 60.0       # %
        pressure = 1013.25    # hPa

    return temp, humidity, pressure
