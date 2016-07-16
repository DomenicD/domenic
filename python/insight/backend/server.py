from flask import Flask, send_from_directory, request, Response, json

from python.notebooks.backpropagation import FeedForward, RectifiedLinearUnit, QuadraticCost, \
    SequenceWeightGenerator

app = Flask(__name__)


@app.route('/feedforward', methods=["POST"])
def feedforward():
    layers = request.json["layers"]
    ff = FeedForward(layers,
                     RectifiedLinearUnit(),
                     QuadraticCost(),
                     SequenceWeightGenerator())
    js = json.dumps([w.tolist() for w in ff.weights])
    return Response(js, status=200, mimetype='application/json')


@app.route('/<path:path>', methods=["GET"])
def static_files(path):
    return send_from_directory('../frontend/bin', path)


if __name__ == "__main__":
    app.run()
