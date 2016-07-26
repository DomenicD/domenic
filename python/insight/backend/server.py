from flask import Flask, send_from_directory, request, Response, json, session

from python.notebooks.networks import SimpleFeedForward, RectifiedLinearUnitActivation, QuadraticCost, \
    SequenceWeightGenerator

app = Flask(__name__)

global_cache = {}


@app.route('/create_feedforward', methods=["POST"])
def create_feedforward():
    layers = request.json["layers"]

    ff = SimpleFeedForward(layers,
                           RectifiedLinearUnitActivation(),
                           QuadraticCost(),
                           SequenceWeightGenerator())
    global_cache[ff.id] = ff
    return create_response(ff.to_web_safe_object())


@app.route('/get_feedforward/<id>', methods=["GET"])
def get_feedforward(id: str) -> SimpleFeedForward:
    ff = _get_feedforward(id)
    return create_response(ff.to_web_safe_object())


@app.route('/update_feedforward/<id>/<command>', methods=["POST"])
def update_feedforward(id: str, command: str):
    ff = _get_feedforward(id)
    args = request.json["args"]
    getattr(ff, command)(*args)
    return create_response(ff.to_web_safe_object())


@app.route('/<path:path>', methods=["GET"])
def static_files(path):
    return send_from_directory('../frontend', path)


def _get_feedforward(id):
    ff = global_cache[id]
    if ff is None or not isinstance(ff, SimpleFeedForward):
        raise ValueError('Invalid id')
    return ff


def create_response(data):
    return Response(json.dumps(data), status=200, mimetype='application/json')


if __name__ == "__main__":
    app.run()
