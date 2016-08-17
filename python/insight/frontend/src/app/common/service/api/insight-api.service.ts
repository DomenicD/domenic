import { Injectable } from '@angular/core';
import {Http, Response} from "@angular/http";
import {Observable} from "rxjs";

@Injectable()
export class InsightApiService {

  constructor(private http: Http) {}

  createNetwork(layers: string[] | number[], type: NetworkType,
                options: Object): Observable<NeuralNetworkDomain> {
    return this
        .postRequestProcessing(this.http.post(
            "/create_network",
            {layers : toNumbers(layers), type : NetworkType[type], options}))
        .then(result => new NeuralNetworkDomain(this, result));
  }

  createTrainer(network: NeuralNetwork, type: TrainerType,
                options: Object): Observable<TrainerDomain> {
    return this.postRequestProcessing(this.http.post(
        "/create_trainer",
        {networkId : network.id, type : TrainerType[type], options}))
        .then(result => new TrainerDomain(this, result));
  }

  networkCommand(networkId: string, command: NetworkCommand,
                 ...args: any[]): Observable<NeuralNetwork> {
    return this.remoteCommand<NeuralNetwork>(networkId, command, args);
  }

  trainerCommand(trainerId: string, command: TrainerCommand,
                 ...args: any[]): Observable<Trainer> {
    return this.remoteCommand<Trainer>(trainerId, command, args);
  }

  private remoteCommand<T>(targetId: string, command: string,
                        ...args: any[]): Observable<T> {
    return this.postRequestProcessing(
        this.http.post(`/remote_command/${targetId}/${command}`, {args}));
  }

  private postRequestProcessing(response: Observable<Response>) {
    return response.map(r => r.json().data);
  }

}
