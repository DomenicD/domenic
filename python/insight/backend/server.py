from flask import Flask, send_from_directory, request, Response, json, session

from python.notebooks.assembled_models import quadratic_feed_forward_network
from python.notebooks.networks import NeuralNetwork
from python.notebooks.serializers import serialize_trainer, serialize_neural_network
from python.notebooks.trainers import ClosedFormFunctionTrainer, Trainer

app = Flask(__name__)

global_cache = {}


@app.route('/create_network', methods=["POST"])
def create_network():
    layers = request.json["layers"]
    network_type = request.json["type"]
    options = request.json["options"]

    if network_type == "QUADRATIC_FEED_FORWARD":
        ff = quadratic_feed_forward_network(
            layers, param_update_rate=options["paramUpdateRate"])
    else:
        raise ValueError(network_type + " is not implemented")

    global_cache[ff.id] = ff
    return create_response(serialize(ff))


@app.route('/create_trainer', methods=["POST"])
def create_trainer():
    network_id = request.json["networkId"]
    trainer_type = request.json["type"]
    options = request.json["options"]

    if trainer_type == "CLOSED_FORM_FUNCTION":
        trainer = ClosedFormFunctionTrainer(
            global_cache[network_id],
            eval(options["function"]),
            options("domain"),
            options("batchSize"))
    else:
        raise ValueError(trainer_type + " is not implemented")

    global_cache[trainer.id] = trainer
    return create_response(serialize(trainer))


def serialize(target):
    if isinstance(target, Trainer):
        return serialize_trainer(target)
    elif isinstance(target, NeuralNetwork):
        return serialize_neural_network(target)
    else:
        raise ValueError("Serialization not implemented for " + type(target).__name__)


@app.route('/remote_command/<target_id>/<command>', methods=["POST"])
def remote_command(target_id: str, command: str):
    target = global_cache[target_id]
    if target is None:
        raise ValueError("No object found with id " + target_id)

    args = request.json["args"]
    getattr(target, command)(*args)
    return create_response(serialize(target))


@app.route('/<path:path>', methods=["GET"])
def static_files(path):
    return send_from_directory('../frontend', path)


def create_response(data):
    return Response(json.dumps(data), status=200, mimetype='application/json')


if __name__ == "__main__":
    app.run()
