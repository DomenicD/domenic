interface FeedForward {
    id: string;
    biases: number[][];
    weights: number[][][];
    inputs: number[][];
    outputs: number[][];
    node_errors: number[][];
    weight_gradients: number[][];
    bias_gradients: number[][];
    total_error: number;
}

class FeedForwardDomain implements FeedForward {
    id: string;
    biases: number[][];
    weights: number[][][];
    inputs: number[][];
    outputs: number[][];
    node_errors: number[][];
    weight_gradients: number[][];
    bias_gradients: number[][];
    total_error: number;

    constructor(private neuralNetworkApi: NeuralNetworkApi, ff: FeedForward) {
        this.updateState(ff);
    }

    forwardPass(inputs: number[]): ng.IPromise<void> {
        // TODO(domenicd): Call the update API method and update this internal state.
    }

    backwardPass(expected: number[]) {

    }

    adjustWeights(learn_rate: number) {

    }

    adjustBiases(learn_rate: number) {

    }

    adjustParameters(learn_rate: number) {

    }

    private updateState(ff: FeedForward) {
        this.id = ff.id;
        this.biases = ff.biases;
        this.weights = ff.weights;
        this.inputs = ff.inputs;
        this.outputs = ff.outputs;
        this.node_errors = ff.node_errors;
        this.weight_gradients = ff.weight_gradients;
        this.bias_gradients = ff.bias_gradients;
        this.total_error = ff.total_error;
    }
}



const FeedForwardCommand = {
    get FORWARD_PASS(): string {
        return 'forward_pass'
    },
    get BACKWARD_PASS(): string {
        return 'backward_pass'
    },
    get ADJUST_WEIGHTS(): string {
        return 'adjust_weights'
    },
    get ADJUST_BIASES(): string {
        return 'adjust_biases'
    },
    get ADJUST_PARAMETERS(): string {
        return 'adjust_parameters'
    }
};

class NeuralNetworkApi {

    constructor(private $http: ng.IHttpService) {
    }

    createFeedForward(layers: number[]): ng.IPromise<FeedForward> {
        return this.postRequestProcessing(this.$http.post("/create_feedforward", { layers: layers }));
    }
    
    getFeedForward(id: string) {
        return this.postRequestProcessing(this.$http.get(`/get_feedforward/${id}`));
    }

    updateFeedForward(id: string, command: )

    
    private postRequestProcessing(response: ng.IPromise<any>) {
        return response.then(r => r.data);
    }
}
class InsightController {
    private feedforward: FeedForward;
    
    constructor(private neuralNetworkApi) {
        this.neuralNetworkApi.createFeedForward([1, 3, 1]).then(ff => this.feedforward = ff);
    }
    
    
           
    update() {
        this.neuralNetworkApi.feedForward([1, 2, 1]).then(function (response) { return this.feedForward = response; });
    }
}

let insight = angular.module('insight', ['ngMaterial'])
    .service('neuralNetworkApi', NeuralNetworkApi)
    .controller('insightController', InsightController);

angular.bootstrap(document, [insight.name]);
