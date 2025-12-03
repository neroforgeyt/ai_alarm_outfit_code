#imports
import pandas as pd  
from chronos import Chronos2Pipeline
import matplotlib.pyplot as plt  
import requests
from meteostat import Stations, Hourly
from datetime import datetime, timedelta
import argparse
import time
import json
import pyttsx3
import pyautogui
import ctypes

# Only needed to trigger video
'''
import subprocess
def record_webcam(duration_seconds=140, output_file="output.mp4"):
    video_device = "Logi C270 HD WebCam"
    audio_device = "Microphone (2- Yeti Stereo Microphone)"
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output file
        "-f", "dshow",
        "-rtbufsize", "100M",
        "-i", f'video={video_device}:audio={audio_device}',
        "-t", str(duration_seconds),
        "-s", "1280x720",     # 720p resolution
        "-r", "30",  # frame rate
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",  # ensures compatibility with Windows players
        "-acodec", "aac",
        "-b:a", "128k",  # audio bitrate
        output_file
    ]
    process = subprocess.Popen(cmd,stdout=subprocess.DEVNULL,)
    '''
####################################

pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")

#Get weather data 
def get_weather_data(lat,lon, startdate, enddate):
    stations = Stations().nearby(lat, lon)
    station = stations.fetch(1)

    if station.empty:
        raise ValueError("No weather station found near the specified location.")
    station_id = station.index[0]

    data = Hourly(station_id, startdate, enddate).fetch()

    if data.empty:
        raise ValueError("No weather data found for the specified location and date range.")
    
    df = data.reset_index()[['time', 'temp']]
    df = df.rename(columns={'time': 'timestamp', 'temp': 'target'})

    df['target'] = df['target'] * 9/5 + 32  # Convert Celsius to Fahrenheit

    # Aggregate to daily max temperature
    df = df.groupby(df['timestamp'].dt.date)['target'].max().reset_index()
    df = df.rename(columns={'timestamp': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    #print(df)
    #input("Press Enter to continue...")
    most_recent_day = df['timestamp'].max().date()
    #print("Most recent day with data:", most_recent_day)
    #print("Max temperature recorded on that day:", df.loc[df['timestamp'].dt.date == most_recent_day, 'target'].values[0])
    return df
    
def format_data(data):
    df = data.copy()
    df['id'] = "Location"
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()

    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_index)

    df['target'] = df['target'].interpolate()# interpolate for any missing datapoints 
    df['id'] = df['id'].fillna('Location')

    df = df.reset_index().rename(columns={'index': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['id', 'timestamp', 'target']]
    
    #print(df)
    #input("Press Enter to continue...")
    return df

def get_dates(days_back):
    enddate = datetime.now() #- pd.Timedelta(days=1)
    startdate = enddate - pd.Timedelta(days=int(days_back))
    print("Fetching data from", startdate.date(), "to", enddate.date())
    return startdate, enddate

def llama_3_output(temp):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3.1:8b",
        "prompt": f'''The predicted temperature for today is {temp} degrees Fahrenheit. 
        Given this list of clothes that I own can you make a recommendation for what I should wear today based on the weather?
        The reccomendation should take into consideration the max temperature and reflect seasonal appropriateness. Please choose a single outfit
        consisting of a shrt, pants, and optionally (for you to choose if I wear or not) a jacket/hoodie and hat/gloves if necessary. Your suggestion must be absolute and
        you are not allowed to leave anything up to my decision. In general err on the side of being less formal.
        Here is the list of clothes I own:
        List: Black hoodie, white hoodie, blue hoodie, blue sweater, purple sweater, red winter coat, jeans, black slacks, 
        blue athletic shorts, black athletic shorts, yellow t-shirt,
        blue sweatpants, black sweatpants, red t-shirt, blue t-shirt, black t-shirt, gray button-up shirt, blue golf shirt, 
        snow boots, tennis shoes, baseball hat, winter hat, gloves''',
        "stream": False
    }
    resp = requests.post(url, json=data, stream=False)
    print(resp.json()["response"])
    return resp.json()["response"]

parser = argparse.ArgumentParser(description="Weather Forecasting with Chronos2")
parser.add_argument("--latitude", type=float, required=True, help="Latitude of the location")
parser.add_argument("--longitude", type=float, required=True, help="Longitude of the location")
parser.add_argument("--previous_days", type=str, default="100", help="Number of previous days for historical data")
parser.add_argument("--time", type=str, required=False, help="Optional: Target time to run the script (HH:MM in 24-hour format)")

if __name__ == "__main__":
    args = parser.parse_args()
    lat = args.latitude
    lon = args.longitude
    if args.time:

        # Target time (today at 7:30 PM)
        target_hour = int(args.time.split(":")[0])
        target_minute = int(args.time.split(":")[1])
        # Get the current time
        now = datetime.now()

        # Create a datetime object for today at the target time
        target_time = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

        # If the target time is already past today, schedule for tomorrow
        if target_time < now:
            target_time += timedelta(days=1)

        # Calculate the number of seconds to wait
        seconds_to_wait = (target_time - now).total_seconds()

        print(f"Waiting for {seconds_to_wait} seconds until {target_time}...")
        time.sleep(seconds_to_wait)  # Pauses execution until target time

    # Simulate key press (Shift) to wake screen
    print("Pressing key to prevent screen lock...")

    ctypes.windll.user32.keybd_event(0x10, 0, 0, 0)  # press Shift
    ctypes.windll.user32.keybd_event(0x10, 0, 2, 0)  # release Shift

    #If wanting to record webcam install ffmpeg and uncomment this section as well as section at top
    #print("Recording webcam video and audio for 2 minutes...")
    #record_webcam()

    previous_days = args.previous_days
    startdate,enddate = get_dates(previous_days)

    data = get_weather_data(lat, lon, startdate, enddate)
    context_df = format_data(data)

    # Generate predictions with covariates
    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=3,  # Number of steps to forecast
        quantile_levels=[0.1, 0.5, 0.9],  # Quantile for probabilistic forecast
        id_column="id",  # Column identifying different time series
        timestamp_column="timestamp",  # Column with datetime information
        target="target",  # Column(s) with time series values to predict
    )
    ts_context = context_df.set_index("timestamp")["target"].tail(256)
    ts_pred = pred_df.set_index("timestamp")

    ts_context.plot(label="historical data", color="xkcd:azure", figsize=(12, 3))
    ts_pred["predictions"].plot(label="forecast", color="xkcd:violet")
    plt.fill_between(
        ts_pred.index,
        ts_pred["0.1"],
        ts_pred["0.9"],
        alpha=0.7,
        label="prediction interval",
        color="xkcd:light lavender",
    )
    first_prediction = pred_df['predictions'].iloc[0]
    print("The predicted temperature for today is:", first_prediction)   
    #plt.legend()
    #plt.show()
    what_should_i_wear = llama_3_output(first_prediction)
    #print("Based on the predicted temperature, you should wear:", what_should_i_wear)
    text = what_should_i_wear.replace("*", "")
    engine = pyttsx3.init()
    engine.say(f"The predicted temperature for today is {first_prediction} degrees Fahrenheit. {text}")
    engine.runAndWait()