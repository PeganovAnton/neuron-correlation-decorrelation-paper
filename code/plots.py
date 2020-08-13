import argparse
import json
import sqlite3

import matplotlib.pyplot as plt

def draw_plots():
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to plot configuration file")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    return config

config = parse_args()
database = config["db_path"]
conn = sqlite3.connect(database)

cur = conn.cursor()



cur.close()
conn.close()
