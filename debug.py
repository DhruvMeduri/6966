import json

ele1 = {"Review":"check1","label":"check2","predicted":"check3"}
ele2 = {"Review":"check4","label":"check5","predicted":"check6"}

with open("output.json", "w") as outfile:
    json.dump(ele1, outfile)
    json.dump(ele2, outfile)