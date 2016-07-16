var NeuralNetworkApi = (function () {
    function NeuralNetworkApi($http) {
        this.$http = $http;
    }
    NeuralNetworkApi.prototype.feedForward = function (layers) {
        return this.$http.post("/feedforward", { layers: layers }).then(function (r) { return r.data; });
    };
    return NeuralNetworkApi;
}());
var InsightController = (function () {
    function InsightController(neuralNetworkApi) {
        this.neuralNetworkApi = neuralNetworkApi;
    }
    InsightController.prototype.getFeedForward = function () {
        var _this = this;
        this.neuralNetworkApi.feedForward([1, 2, 1]).then(function (response) { return _this.feedForward = response; });
    };
    return InsightController;
}());
var insight = angular.module('insight', ['ngMaterial'])
    .service('neuralNetworkApi', NeuralNetworkApi)
    .controller('insightController', InsightController);
angular.bootstrap(document, [insight.name]);
