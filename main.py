from src.weather import weather_data
from src.text_generator import generate_text

logging.basicConfig(level=logging.INFO)
logging.info('Initiating Weather data generator')

def main():
    parser = argparse.ArgumentParser(description='Pull weather data to modulate audio signals')
    parser.add_argument('--city', dest='city', default='London', type=str, help='Name of the city for which to retrieve weather data, which impacts the text generation temp')
    parser.add_argument('--prompt', dest='prompt', default="Welcome to Seasonal Generation Radio, today we're going to be talking about", type=str, help='Starting prompt which will trigger the text generation')
    parser.add_argument('--max_length', dest='max_length', default='250', type=str, help='Size of the generated text in tokens')

    args = parser.parse_args()

    weather = weather_data(args.city)
    temp = weather.get('Temperature Min') / weather.get('Temperature Max')
    text = generate_text(args.prompt, temp, args.max_length)
    # audio = tts(text)

if __name__ == "__main__":
    main()
