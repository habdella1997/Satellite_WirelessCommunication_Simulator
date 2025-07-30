import requests
import simulator.constants as C


def get_weather_stats(lat, long):
    payload = {'lat': str(lat), 'lon': str(long), 'exclude': 'minutely,hourly,daily,alerts', 'appid': C.WEATHER_API_KEY}
    r = requests.get('https://api.openweathermap.org/data/2.5/onecall', params=payload)
    json_str = r.json()['current']
    temp = json_str['temp']
    humidty = json_str['humidity']
    pressure = json_str['pressure']
    return float(temp), float(humidty), float(pressure)
