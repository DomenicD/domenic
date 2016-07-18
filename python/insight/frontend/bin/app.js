function toNumbers(list) {
    return list.map(function (str) { return Number(str); });
}
function toNumber(value) { return Number(value); }
var FeedForwardDomain = (function () {
    function FeedForwardDomain(neuralNetworkApi, ff) {
        this.neuralNetworkApi = neuralNetworkApi;
        this.updateState(ff);
    }
    FeedForwardDomain.prototype.forwardPass = function (inputs) {
        var _this = this;
        return this.neuralNetworkApi
            .updateFeedForward(this.id, FeedForwardCommandEnum.FORWARD_PASS, toNumbers(inputs))
            .then(function (ff) { return _this.updateState(ff); });
    };
    FeedForwardDomain.prototype.backwardPass = function (expected) {
        var _this = this;
        return this.neuralNetworkApi
            .updateFeedForward(this.id, FeedForwardCommandEnum.BACKWARD_PASS, toNumbers(expected))
            .then(function (ff) { return _this.updateState(ff); });
    };
    FeedForwardDomain.prototype.adjustWeights = function (learningRate) {
        var _this = this;
        return this.neuralNetworkApi
            .updateFeedForward(this.id, FeedForwardCommandEnum.ADJUST_WEIGHTS, toNumber(learningRate))
            .then(function (ff) { return _this.updateState(ff); });
    };
    FeedForwardDomain.prototype.adjustBiases = function (learningRate) {
        var _this = this;
        return this.neuralNetworkApi
            .updateFeedForward(this.id, FeedForwardCommandEnum.ADJUST_BIASES, toNumber(learningRate))
            .then(function (ff) { return _this.updateState(ff); });
    };
    FeedForwardDomain.prototype.adjustParameters = function (learningRate) {
        var _this = this;
        return this.neuralNetworkApi
            .updateFeedForward(this.id, FeedForwardCommandEnum.ADJUST_PARAMETERS, toNumber(learningRate))
            .then(function (ff) { return _this.updateState(ff); });
    };
    FeedForwardDomain.prototype.updateState = function (ff) {
        this.id = ff.id;
        this.biases = ff.biases;
        this.weights = ff.weights;
        this.inputs = ff.inputs;
        this.outputs = ff.outputs;
        this.node_errors = ff.node_errors;
        this.weight_gradients = ff.weight_gradients;
        this.bias_gradients = ff.bias_gradients;
        this.total_error = ff.total_error;
    };
    return FeedForwardDomain;
}());
var FeedForwardCommandEnum = {
    get FORWARD_PASS() { return 'forward_pass'; },
    get BACKWARD_PASS() { return 'backward_pass'; },
    get ADJUST_WEIGHTS() { return 'adjust_weights'; },
    get ADJUST_BIASES() { return 'adjust_biases'; },
    get ADJUST_PARAMETERS() { return 'adjust_parameters'; }
};
var NeuralNetworkApi = (function () {
    function NeuralNetworkApi($http) {
        this.$http = $http;
    }
    NeuralNetworkApi.prototype.createFeedForward = function (layers) {
        var _this = this;
        return this
            .postRequestProcessing(this.$http.post("/create_feedforward", { layers: toNumbers(layers) }))
            .then(function (ff) { return new FeedForwardDomain(_this, ff); });
    };
    NeuralNetworkApi.prototype.getFeedForward = function (id) {
        var _this = this;
        return this.postRequestProcessing(this.$http.get("/get_feedforward/" + id))
            .then(function (ff) { return new FeedForwardDomain(_this, ff); });
    };
    NeuralNetworkApi.prototype.updateFeedForward = function (id, command) {
        var args = [];
        for (var _i = 2; _i < arguments.length; _i++) {
            args[_i - 2] = arguments[_i];
        }
        return this.postRequestProcessing(this.$http.post("/update_feedforward/" + id + "/" + command, { args: args }));
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
        this.neuralNetworkApi.createFeedForward(layers).then(function (ff) { return _this.feedForward = ff; });
    };
    return InsightController;
}());
var insight = angular.module('insight', ['ngMaterial'])
    .service('neuralNetworkApi', NeuralNetworkApi)
    .controller('insightController', InsightController);
angular.bootstrap(document, [insight.name]);
//# sourceMappingURL=app.js.map