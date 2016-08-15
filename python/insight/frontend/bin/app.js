var __extends = (this && this.__extends) || function (d, b) {
    for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p];
    function __() { this.constructor = d; }
    d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
};
function toNumbers(list) {
    return list.map(function (str) { return Number(str); });
}
function toNumber(value) { return Number(value); }
var DomainObject = (function () {
    function DomainObject(neuralNetworkApi, response) {
        this.neuralNetworkApi = neuralNetworkApi;
        this.response = response;
    }
    DomainObject.prototype.postRequestProcessing = function (responsePromise) {
        var _this = this;
        return responsePromise.then(function (response) {
            _this.response = response;
        });
    };
    return DomainObject;
}());
var NeuralNetworkDomain = (function (_super) {
    __extends(NeuralNetworkDomain, _super);
    function NeuralNetworkDomain(neuralNetworkApi, response) {
        _super.call(this, neuralNetworkApi, response);
    }
    Object.defineProperty(NeuralNetworkDomain.prototype, "id", {
        get: function () {
            return this.response.id;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(NeuralNetworkDomain.prototype, "totalError", {
        get: function () {
            return this.response.totalError;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(NeuralNetworkDomain.prototype, "inputCount", {
        get: function () {
            return this.response.inputCount;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(NeuralNetworkDomain.prototype, "outputCount", {
        get: function () {
            return this.response.outputCount;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(NeuralNetworkDomain.prototype, "layerCount", {
        get: function () {
            return this.response.layerCount;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(NeuralNetworkDomain.prototype, "parameters", {
        get: function () {
            return this.response.parameters;
        },
        enumerable: true,
        configurable: true
    });
    NeuralNetworkDomain.prototype.forwardPass = function (inputs) {
        return this.postRequestProcessing(this.neuralNetworkApi.networkCommand(this.id, "forward_pass", toNumbers(inputs)));
    };
    NeuralNetworkDomain.prototype.backwardPass = function (expected) {
        return this.postRequestProcessing(this.neuralNetworkApi.networkCommand(this.id, "backward_pass", toNumbers(expected)));
    };
    return NeuralNetworkDomain;
}(DomainObject));
var TrainerDomain = (function (_super) {
    __extends(TrainerDomain, _super);
    function TrainerDomain(neuralNetworkApi, response) {
        _super.call(this, neuralNetworkApi, response);
    }
    Object.defineProperty(TrainerDomain.prototype, "id", {
        get: function () {
            return this.response.id;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TrainerDomain.prototype, "networkId", {
        get: function () {
            return this.response.networkId;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TrainerDomain.prototype, "batchSize", {
        get: function () {
            return this.response.batchSize;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TrainerDomain.prototype, "stepTally", {
        get: function () {
            return this.response.stepTally;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TrainerDomain.prototype, "batchTally", {
        get: function () {
            return this.response.batchTally;
        },
        enumerable: true,
        configurable: true
    });
    Object.defineProperty(TrainerDomain.prototype, "batchResults", {
        get: function () {
            return this.response.batchResults;
        },
        enumerable: true,
        configurable: true
    });
    return TrainerDomain;
}(DomainObject));
var NetworkType;
(function (NetworkType) {
    NetworkType[NetworkType["STANDARD_FEED_FORWARD"] = 0] = "STANDARD_FEED_FORWARD";
    NetworkType[NetworkType["QUADRATIC_FEED_FORWARD"] = 1] = "QUADRATIC_FEED_FORWARD";
})(NetworkType || (NetworkType = {}));
var TrainerType;
(function (TrainerType) {
    TrainerType[TrainerType["CLOSED_FORM_FUNCTION"] = 0] = "CLOSED_FORM_FUNCTION";
})(TrainerType || (TrainerType = {}));
var NeuralNetworkApi = (function () {
    function NeuralNetworkApi($http) {
        this.$http = $http;
    }
    NeuralNetworkApi.prototype.createNetwork = function (type, layers, options) {
        var _this = this;
        return this.postRequestProcessing(this.$http.post("/create_network", { layers: toNumbers(layers), type: NetworkType[type], options: options }))
            .then(function (result) { return new NeuralNetworkDomain(_this, result); });
    };
    NeuralNetworkApi.prototype.createTrainer = function (network, type, options) {
        return this.postRequestProcessing(this.$http.post("/create_trainer", { networkId: network.id, type: TrainerType[type], options: options }));
    };
    NeuralNetworkApi.prototype.networkCommand = function (networkId, command) {
        var args = [];
        for (var _i = 2; _i < arguments.length; _i++) {
            args[_i - 2] = arguments[_i];
        }
        return this.remoteCommand(networkId, command, args);
    };
    NeuralNetworkApi.prototype.trainerCommand = function (trainerId, command) {
        var args = [];
        for (var _i = 2; _i < arguments.length; _i++) {
            args[_i - 2] = arguments[_i];
        }
        return this.remoteCommand(trainerId, command, args);
    };
    NeuralNetworkApi.prototype.remoteCommand = function (targetId, command) {
        var args = [];
        for (var _i = 2; _i < arguments.length; _i++) {
            args[_i - 2] = arguments[_i];
        }
        return this.postRequestProcessing(this.$http.post("/remote_command/" + targetId + "/" + command, { args: args }));
    };
    NeuralNetworkApi.prototype.postRequestProcessing = function (response) {
        return response.then(function (r) { return r.data; });
    };
    return NeuralNetworkApi;
}());
var InsightController = (function () {
    function InsightController(neuralNetworkApi) {
        this.neuralNetworkApi = neuralNetworkApi;
    }
    InsightController.prototype.createFeedForward = function (layers) {
        var _this = this;
        this.neuralNetworkApi.createNetwork(layers).then(function (ff) { return _this.feedForward = ff; });
    };
    return InsightController;
}());
var insight = angular.module('insight', ['ngMaterial'])
    .service('neuralNetworkApi', NeuralNetworkApi)
    .controller('insightController', InsightController);
angular.bootstrap(document, [insight.name]);
//# sourceMappingURL=app.js.map