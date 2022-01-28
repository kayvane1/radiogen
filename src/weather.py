import pandas as pd
import requests

logging.basicConfig(level=logging.INFO)
logging.info('Initiating Weather data generator')

def main():
    parser = argparse.ArgumentParser(description='Pull weather data to modulate audio signals')
    parser.add_argument('--city', dest='city', type=str, help='Name of the city for which to retrieve data')
    parser.add_argument('--stream', dest='stream', default='no', type=str, help='options: yes / no for whether a streaming dataframe should be created')
    parser.add_argument('--refresh', dest='refresh', default='30s', type=str, help='options: 1s / 10s / 30s at what frequency the dataframe should be updated')

    args = parser.parse_args()


    if args.stream == 'no':
        return pd.DataFrame(weather_data(city), index=[0])
    elif args.stream == 'yes' and args.refresh in ['1s','10s','30s']:
        logging.info(f'Data Stream initiated at a refresh interval of {args.refresh}')
        return PeriodicDataFrame(streaming_weather_data, interval=args.refresh)
    else:
        logging.info('Error in input parameters')
        print("please provide a valid combination of arguments, see help menu (-h) for usage instructions")

def weather_data(city, openweathermap_api_key=openweathermap_api_key):
    """
    Get weather data for a list of cities using the openweathermap API
    parameters:
    city(str): Name of city from which current data is fetched
    """
    data = {}
    res = requests.get(f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={openweathermap_api_key}&units=metric')
    weather = res.json()
    data['city'] = args.city
    data['Lat'] = weather.get('coord',{}).get('lat',0)
    data['Lon'] = weather.get('coord',{}).get('lon',0)
    data['Temperature'] = weather.get('main',{}).get('temp',0) # Temperature. Unit Default: Kelvin, Metric: Celsius
    data['Temperature Max'] = weather.get('main',{}).get('temp_max',0) # Maximum temperature at the moment. This is maximal currently observed temperature (within large megalopolises and urban areas).
    data['Temperature Min'] = weather.get('main',{}).get('temp_min',0) # Minimum temperature at the moment. This is minimal currently observed temperature (within large megalopolises and urban areas)
    data['Feels Like'] = weather.get('main',{}).get('feels_like',0) # Temperature. This temperature parameter accounts for the human perception of weather.
    data['Visibility'] = weather.get('visibility',0) # Visibility, meter
    data['Humidity'] = weather.get('main',{}).get('humidity',0) # Humidity, %
    data['Pressure'] = weather.get('main',{}).get('pressure',0) # Atmospheric pressure (on the sea level, if there is no sea_level or grnd_level data), hPa
    data['Wind Speed'] = weather.get('wind',{}).get('speed',0) # Wind speed. Unit Default: meter/sec, Metric: meter/sec,
    data['Wind Gust'] = weather.get('main',{}).get('gust',0) # Wind gust. Unit Default: meter/sec, Metric: meter/sec, Imperial: miles/hour
    data['Wind Deg'] = weather.get('clouds',{}).get('deg',0) # Wind direction, degrees (meteorological)
    data['Clouds'] = weather.get('clouds',{}).get('all',0) # Cloudiness, %
    data['Snow 1h'] = weather.get('snow',{}).get('1h',0) # Rain volume for the last 1 hour, mm
    data['Snow 3h'] = weather.get('snow',{}).get('3h',0) # Rain volume for the last 3 hours, mm
    data['Rain 1h'] = weather.get('rain',{}).get('1h',0) #  Snow volume for the last 1 hour, mm
    data['Rain 3h'] = weather.get('rain',{}).get('3h',0) # Snow volume for the last 3 hours, mm
    data['weather'] = weather.get('weather',{})[0].get('main',0) # Group of weather parameters (Rain, Snow, Extreme etc.)
    data['weather_desc'] = weather.get('weather',{})[0].get('description',0) # Weather condition within the group.
    data['time'] = pd.Timestamp.now()
    return data

def streaming_weather_data(**kwargs):
    """
    callback function
    get London weather data
    """
    df = pd.DataFrame(weather_data(args.city), index=[0])
    return df.set_index('time')

if __name__ == "__main__":
    main()
