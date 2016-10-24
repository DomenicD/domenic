from flask import Flask, send_from_directory, request, Response, json
from flask_cors import CORS

import modeling.assembled_models as am
from modeling.common.serializers import serialize
from modeling.layers import QuadraticLayer, LinearLayer
from modeling.trainers import ClosedFormFunctionTrainer

app = Flask(__name__)
CORS(app)

global_cache = {}


@app.route('/updater_keys', methods=["GET"])
def updater_keys():
    return create_response([key for key in am.updaters.keys()])


@app.route('/create_network', methods=["POST"])
def create_network():
    layers = request.json["layers"]
    network_type = request.json["type"]
    options = request.json["options"]

    if network_type == "QUADRATIC_FEED_FORWARD":
        network = am.feed_forward_network(QuadraticLayer, layers, options["updater"])
    elif network_type == "STANDARD_FEED_FORWARD":
        network = am.feed_forward_network(LinearLayer, layers, options["updater"])
    else:
        raise ValueError(network_type + " is not implemented")

    global_cache[network.id] = network
    return create_response(serialize(network))


@app.route('/create_trainer', methods=["POST"])
def create_trainer():
    network_id = request.json["networkId"]
    trainer_type = request.json["type"]
    options = request.json["options"]

    if trainer_type == "CLOSED_FORM_FUNCTION":
        trainer = ClosedFormFunctionTrainer(
            global_cache[network_id],
            eval(options["function"]),
            options["domain"],
            options["batchSize"])
    else:
        raise ValueError(trainer_type + " is not implemented")

    global_cache[trainer.id] = trainer
    return create_response(serialize(trainer))


@app.route('/remote_command/<target_id>/<command>', methods=["POST"])
def remote_command(target_id: str, command: str):
    target = global_cache[target_id]
    if target is None:
        raise ValueError("No object found with id " + target_id)

    args = request.json["args"]
    return create_response(serialize(getattr(target, command)(*args)))


@app.route('/<path:path>', methods=["GET"])
def static_files(path):
    return send_from_directory('../frontend', path)


def create_response(data):
    return Response(json.dumps(data), status=200, mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0')
