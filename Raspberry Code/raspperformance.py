import os
import psutil
import time
import datetime



file=open("/home/pi/Downloads/rasp_per.csv", "a")
if os.stat("/home/pi/Downloads/rasp_per.csv").st_size==0:
    file.write("Time, CPU Usage, CPU Temp, \n")

while True:
    
    now=datetime.datetime.now()
    usage=psutil.cpu_percent()
   
    temp=psutil.sensors_temperatures()
    
    
    file.write(str(now)+","+str(usage)+","+str(temp)+"\n")

    #print(str(now)+ " ", "CPU_usage(%)", usage, " ",freq, "", "CPU_temp",temp, "\n")
    file.flush()
    time.sleep(10)

file.close()