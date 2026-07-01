#!/usr/bin/env python3
import os
import glob
import json

def get_temp():
    temps = {}
    for tz in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            with open(os.path.join(tz, "type"), "r") as f:
                name = f.read().strip()
            with open(os.path.join(tz, "temp"), "r") as f:
                temp = int(f.read().strip()) / 1000.0
            temps[f"temp_{name}"] = temp
        except:
            pass
    return temps

def get_freq():
    freqs = {}
    for cpu in glob.glob("/sys/devices/system/cpu/cpu[0-9]*"):
        try:
            cpu_id = os.path.basename(cpu)
            with open(os.path.join(cpu, "cpufreq/scaling_cur_freq"), "r") as f:
                freq = int(f.read().strip()) / 1000.0  # MHz
            freqs[f"freq_{cpu_id}"] = freq
        except:
            pass
    return freqs

def main():
    vitals = {}
    vitals.update(get_temp())
    vitals.update(get_freq())
    
    # Output as JSON for robust parsing
    print(json.dumps(vitals))

if __name__ == "__main__":
    main()
