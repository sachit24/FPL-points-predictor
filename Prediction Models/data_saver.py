import os
import json
import requests

os.makedirs("25-26 data", exist_ok=True)

# Bootstrap static
data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
with open("25-26 data/bootstrap_static.json", "w") as f:
    json.dump(data, f, indent=2)
print("Saved bootstrap_static.json")

# Gameweeks
finished_gws = [gw["id"] for gw in data["events"] if gw["finished"]]
for gw_id in finished_gws:
    gw_data = requests.get(f"https://fantasy.premierleague.com/api/event/{gw_id}/live/").json()
    with open(f"25-26 data/gw{gw_id}.json", "w") as f:
        json.dump(gw_data, f, indent=2)
    print(f"Saved gw{gw_id}.json")