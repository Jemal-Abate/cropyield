import os
import sys
import time
import json
import urllib3
import requests
import multiprocessing
import pandas as pd

urllib3.disable_warnings()

# Destination folder to save the data
dest_dir = './2_Data/WeatherData/'
# File path containing location data of districts with their latitudes and longitudes
locations_file_path = './districtlevel_locations.csv'  # Modify this CSV file to have district locations
# File path for the parameter to download from NASA website, add or remove parameters from this file 
parameters_file_path = './params.csv'
start_date = '20040101'  # Format YYYYMMDD
end_date = '20241220'    # Format YYYYMMDD

def download_function(collection):
    request, filepath, row = collection
    response = requests.get(url=request, verify=False, timeout=30.00).json()

    if 'properties' in response and 'parameter' in response['properties']:
        data = response['properties']['parameter']
        df = pd.DataFrame(data)
        # Generate the date range from '2004-01-01' to '2017-12-30'
        date_range = pd.date_range(start='2004-01-01', end='2024-12-20')
        df['date'] = date_range[:len(df)]  # Ensure the date range matches the DataFrame length
        df['region'] = row['Region']
        df['zone'] = row['Zone']
        df['district'] = row['district']
        df['weredacode'] = row['WeredaCode']
        df.to_csv(filepath, index=False)
    else:
        print(f"Unexpected response format for request: {request}")

class Process:
    def __init__(self):
        self.processes = 5  # Please do not go more than five concurrent requests.
        self.request_template = (
            r"https://power.larc.nasa.gov/api/temporal/daily/point?"
            "parameters={parameters}&community=AG&longitude={longitude}&latitude={latitude}"
            "&start={start_date}&end={end_date}&format=JSON&time-standard=UTC"
        )
        # self.filename_template = "{dir}{loc_name}_Lat_{latitude}_Lon_{longitude}.csv"
        self.filename_template = "{dir}{loc_name}.csv"
        # self.filename_template = "{dir}{loc_name}.csv"
        self.times = {}

    def execute(self, dir, locations_file_path, parameters_file_path, start_date, end_date):
        locations = pd.read_csv(locations_file_path)
        parameters = ','.join(pd.read_csv(parameters_file_path)['Param_Code'].tolist())
        dir = dest_dir
        start_time = time.time()

        requests_list = []
        address=[]
        for _, row in locations.iterrows():
            loc_name = row['district']
            WeredaCode = row['WeredaCode']
            Zone = row['Zone']
            Region = row['Region']
            latitude, longitude = row['lat'], row['lng']
            request = self.request_template.format(
                parameters=parameters, latitude=latitude, longitude=longitude, start_date=start_date, end_date=end_date
            )
            # filename = self.filename_template.format(dir=dir, loc_name=loc_name, latitude=latitude, longitude=longitude)
            filename = self.filename_template.format(dir=dir, loc_name=loc_name, WeredaCode=WeredaCode, Zone=Zone, Region=Region)
            requests_list.append((request, filename, row))

        requests_total = len(requests_list)

        pool = multiprocessing.Pool(self.processes)
        x = pool.imap_unordered(download_function, requests_list)

        for i, _ in enumerate(x, 1):
            sys.stderr.write(f'\rExporting {i/requests_total:.2%}')

        self.times["Total Script"] = round((time.time() - start_time), 2)

        print("\n")
        print("Total Script Time:", self.times["Total Script"])

def main():
    os.makedirs(dest_dir, exist_ok=True)

    # District data with lat and long
    Process().execute(
        dir=dest_dir,
        locations_file_path=locations_file_path,
        parameters_file_path=parameters_file_path,
        start_date=start_date,
        end_date=end_date
    )

if __name__ == "__main__":
    main()
