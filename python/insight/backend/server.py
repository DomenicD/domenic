from flask import Flask, send_from_directory, request, Response, json, session

from python.notebooks.backpropagation import FeedForward, RectifiedLinearUnit, QuadraticCost, \
    SequenceWeightGenerator

app = Flask(__name__)

global_cache = {}

@app.route('/create_feedforward', methods=["POST"])
def create_feedforward():
    layers = request.json["layers"]

    ff = FeedForward(layers,
                     RectifiedLinearUnit(),
                     QuadraticCost(),
                     SequenceWeightGenerator())
    global_cache[ff.id] = ff
    return create_response(ff)



@app.route('/get_feedforward/<id>', methods=["GET"])
def get_feedforward(id: str) -> FeedForward:
    ff = global_cache[id]
    if ff is None or not isinstance(ff, FeedForward):
        raise ValueError('Invalid id')
    return create_response(ff)

@app.route('/update_feedforward/<id>/<command>', methods=["POST"])
def update_feedforward(id: str, command: str):
    ff = get_feedforward(id)
    args = request.json["args"]
    apply_args(getattr(ff, command), args)
    return ff


@app.route('/<path:path>', methods=["GET"])
def static_files(path):
    return send_from_directory('../frontend/bin', path)

def apply_args(func, *args):
    return func(args)

def create_response(data):
    return Response(json.dumps(data), status=200, mimetype='application/json')


if __name__ == "__main__":
    app.run()
