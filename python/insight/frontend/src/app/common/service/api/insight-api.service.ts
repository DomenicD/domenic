import {Injectable, Inject} from '@angular/core';
import {Http, Response} from "@angular/http";
import {Observable} from "rxjs/Rx";
import {
  NetworkType,
  NeuralNetwork,
  TrainerType,
  NetworkCommand,
  TrainerCommand
} from "./insight-api-message";
import {NeuralNetworkDomain} from "../../domain/neural-network";
import {toNumbers} from "../../util/parse";
import {TrainerDomain} from "../../domain/trainer";
import {ApiUrl} from "../../../app.annotations";

@Injectable()
export class InsightApiService {

  constructor(@Inject(ApiUrl) private apiUrl: string, private http: Http) {}

  updaterKeys(): Observable<string[]> {
    return this.postRequestProcessing(this.http.get(this.url('updater_keys')));
  }

  createNetwork(layers: string[] | number[], type: NetworkType,
                options: Object): Observable<NeuralNetworkDomain> {
    return this
        .postRequestProcessing(this.http.post(
            this.url('create_network'),
            {layers : toNumbers(layers), type : NetworkType[type], options}))
        .map(result => new NeuralNetworkDomain(this, result));
  }

  createTrainer(network: NeuralNetwork, type: TrainerType,
                options: Object): Observable<TrainerDomain> {
    return this
        .postRequestProcessing(this.http.post(
            this.url('create_trainer'),
            {networkId : network.id, type : TrainerType[type], options}))
        .map(result => new TrainerDomain(this, result));
  }

  networkCommand<T>(networkId: string, command: NetworkCommand,
                 ...args: any[]): Observable<T> {
    return this.remoteCommand<T>(networkId, command, args);
  }

  trainerCommand<T>(trainerId: string, command: TrainerCommand,
                 ...args: any[]): Observable<T> {
    return this.remoteCommand<T>(trainerId, command, args);
  }

  private remoteCommand<T>(targetId: string, command: string,
                           args: any[]): Observable<T> {
    return this.postRequestProcessing(
        this.http.post(this.url('remote_command', targetId, command), {args}));
  }

  private postRequestProcessing(response: Observable<Response>) {
    return Observable.from(response.map(r => r.json()));
  }

  private url(...path: string[]): string {
    return `${this.apiUrl}/${path.join('/')}`
  }
}
