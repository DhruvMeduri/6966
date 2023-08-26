import json

ele = {"Review":"check1","label":"check2","predicted":"check3"}
with open("output.json", "w") as outfile:
    json.dump(ele, outfile)