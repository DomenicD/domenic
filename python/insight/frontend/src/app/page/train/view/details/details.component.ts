import {Component, OnInit, Input, ViewEncapsulation} from '@angular/core';
import {Subscription} from "rxjs/Rx";
import {TrainerBatchResult} from "../../../../common/service/api/insight-api-message";
import {TrainerDomain} from "../../../../common/domain/trainer";
import {HeatMap, HeatMapComponent} from "../../../../common/component/heat-map/heat-map.component";

@Component({
  moduleId : module.id,
  selector : 'app-details',
  templateUrl : 'details.component.html',
  styleUrls : [ 'details.component.css' ],
  directives: [HeatMapComponent],
  encapsulation : ViewEncapsulation.Native
})
export class DetailsComponent implements OnInit {

  cachedHeatMaps: HeatMap[];
  heatMaps: Map<string, HeatMap> = new Map<string, HeatMap>();
  heatMapHistory: number = 10;

  private _trainer: TrainerDomain;
  private batchResultSubscription: Subscription;

  constructor() {}

  ngOnInit() {}

  @Input()
  get trainer(): TrainerDomain {
    return this._trainer;
  }

  set trainer(value: TrainerDomain) {
    this._trainer = value;
    if (this.batchResultSubscription != null) {
      this.batchResultSubscription.unsubscribe();
    }
    this.batchResultSubscription = this.trainer.onBatchResult.subscribe(
        (batchResult: TrainerBatchResult) =>
            this.updateParameters(batchResult));
  }

  private updateParameters(batchResult: TrainerBatchResult) {
    let paramSetMaps = batchResult.parameters;
    for (let paramSetMap of paramSetMaps) {
      for (let key in paramSetMap) {
        let paramSet = paramSetMap[key];
        let name = paramSet.name;
        let deltas = [].concat(...paramSet.deltas);
        if (!this.heatMaps.has(name)) {
          this.heatMaps.set(name, new HeatMap(name, this.heatMapHistory, deltas.length));
        }
        var heatMap = this.heatMaps.get(name);
        heatMap.addValues(deltas);
      }
    }
    // Convert the Map to an Array.
    this.cachedHeatMaps = Array.from(this.heatMaps.values());
    // Find the global max and min.
    let max = Number.NEGATIVE_INFINITY;
    let min = Number.POSITIVE_INFINITY;
    for (let heatMap of this.cachedHeatMaps) {
      if (heatMap.groupMax > max) {
        max = heatMap.groupMax;
      }
      if (heatMap.groupMin < min) {
        min = heatMap.groupMin;
      }
    }
    // Set the global max and min and update the HeatMap.
    for (let heatMap of this.cachedHeatMaps) {
      heatMap.globalMax = max;
      heatMap.globalMin = min;
      heatMap.update();
    }
  }
}
