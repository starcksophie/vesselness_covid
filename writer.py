import json
import analytics
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    # https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    Special json encoder for numpy types.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def write_result(filename):
    """
    Writes result json to file.

    filename: string: name for outfile.
    """
    json_f = json.dumps(analytics.result, indent = 4, cls=NumpyEncoder)
    # si on recupere que stdout
    print(json_f)
    # si on peut recuperer un fichier
    with open(filename, 'w') as file:
        file.write(json_f)

