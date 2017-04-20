__author__ = 'smujoo'

import requests
import json
import random
import time
import datetime
import math


def create_activity(request_id,source_id,child_id,activity_type,activity_duration,caregiver_id,recorded_time, event_time):

    pa_activity_ep = "https://3zqz1tgu92.execute-api.us-west-2.amazonaws.com/dev"
    pa_activity_api_key = <redacted>


    source_type = "amazon"



    url = "%s/events" %(pa_activity_ep)
    headers = { "request_id" : str(request_id), 'Content-Type' : 'application/json', 'api_key' : pa_activity_api_key}
    data = {}
    data['child_id'] = child_id
    data['type'] = activity_type
    data['caregiver_id'] = caregiver_id
    data['source_id'] = source_id
    data['source_type'] = source_type
    data['event_time'] = event_time
    data['metadata'] = {}
    data['metadata']['duration'] = str(activity_duration)
#    data['metadata']['recorded_time'] = str(recorded_time)
    payload = json.dumps(data)
    response = requests.request("POST", url, data=payload, headers=headers, timeout=5)
    return response.text

def isPrime(num):
     if num < 2:
         return False
     for i in range(2, int(math.sqrt(num)) + 1):
         if num % i == 0:
             return False
         return True

caregiver_id = "58a40a921d90110001fd52b9"
child_id = "58a40a921d90110001fd52ba"
epoch_time = 1492673099.0

print epoch_time

for i in range(0, 500):
    timedelta_in_hrs = random.randrange(1,5)
    timedelta_in_min = random.randrange(1,60)
    timedelta_in_sec = random.randrange(1,60)
    #print "Time Delta: %s" %(timedelta_in_hrs)
    new_time = datetime.datetime.fromtimestamp(epoch_time) + datetime.timedelta(hours=timedelta_in_hrs, minutes=timedelta_in_min, seconds=timedelta_in_sec)
    epoch_time = time.mktime(new_time.timetuple())
    #print "Epoch: %s" %(epoch_time)

    if isPrime(i):
        sleep_duration_in_hrs = random.randrange(3,5)
    else:
        sleep_duration_in_hrs = random.randrange(6,8)

    print "Sleep duration: %s" %(sleep_duration_in_hrs)
    print "===="
    event_time = epoch_time
    activity_duration = sleep_duration_in_hrs
    recorded_time = new_time.time()
    #print recorded_time
    request_id = "1234"
    source_id = "amzn1.ask.account."
    activity_type = "sleep"
    print create_activity(request_id,source_id,child_id,activity_type,activity_duration,caregiver_id,recorded_time, event_time)