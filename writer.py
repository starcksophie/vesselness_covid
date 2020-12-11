import json
import analytics

def write_result():
    json_f = json.dumbs(analytics.result, indent = 4)
    # si on recupere que stdout
    print(json_f)
    # si on peut recuperer un fichier
    """
    with open("out.json") as file:
        file.write(json_f)
    """
