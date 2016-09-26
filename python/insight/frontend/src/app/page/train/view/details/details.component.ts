import {Component, OnInit, Input, ViewEncapsulation} from '@angular/core';
import {Subscription} from "rxjs";
import {
  TrainerBatchResult, ParameterSetMap, ParameterSet, Delta
} from "../../../../common/service/api/insight-api-message";
import {TrainerDomain} from "../../../../common/domain/trainer";
import {
  HeatMap,
  HeatMapComponent,
  HeatMapMode
} from "../../../../common/component/heat-map/heat-map.component";
import {UiFriendlyEnum} from "../../../../common/domain/ui-friendly-enum";
import {PolymerElement} from "@vaadin/angular2-polymer";

const TABS = [ "Deltas", "Gradients", "Weights" ];

@Component({
  moduleId : module.id,
  selector : 'app-details',
  templateUrl : 'details.component.html',
  styleUrls : [ 'details.component.css' ],
  directives : [
    HeatMapComponent,
    PolymerElement('paper-radio-button'),
    PolymerElement('paper-radio-group'),
    PolymerElement('paper-tabs'),
    PolymerElement('paper-tab'),
  ],
  encapsulation : ViewEncapsulation.Native
})
export class DetailsComponent implements OnInit {

  cachedDeltaHeatMaps: HeatMap[] = [];
  deltaHeatMaps: Map<string, HeatMap> = new Map<string, HeatMap>();
  cachedGradientHeatMaps: HeatMap[] = [];
  gradientHeatMaps: Map<string, HeatMap> = new Map<string, HeatMap>();
  heatMapHistory: number = 100;
  tabIndex: number = 0;
  heatMapMode: UiFriendlyEnum<HeatMapMode> =
      new UiFriendlyEnum<HeatMapMode>(HeatMapMode);

  private _trainer: TrainerDomain;
  private batchResultSubscription: Subscription;

  constructor() {}

  ngOnInit() {}

  get tabs(): string[] { return TABS; }

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

  isTabActive(tab: string): boolean {
    return this.tabs.indexOf(tab) === this.tabIndex;
  }

  private updateParameters(batchResult: TrainerBatchResult) {
    let paramSetMaps = batchResult.parameters;
    for (let paramSetMap of paramSetMaps) {
      for (let key in paramSetMap) {
        let paramSet = paramSetMap[key];
        let name = paramSet.name;
        let deltas = [].concat(...paramSet.deltas).map(d => d.value);
        let gradients = [].concat(...paramSet.gradients);
        let weights = [].concat(...paramSet.values);

        this.getHeatMapDefault(name, deltas.length, this.deltaHeatMaps,
                               this.cachedDeltaHeatMaps)
            .addValues(deltas);
        this.getHeatMapDefault(name, gradients.length, this.gradientHeatMaps,
                               this.cachedGradientHeatMaps)
            .addValues(gradients);
      }
    }
    this.updateGlobalMaxMin(this.cachedDeltaHeatMaps);
    this.updateGlobalMaxMin(this.cachedGradientHeatMaps);
  }

  private updateGlobalMaxMin(heatMaps: HeatMap[]) {
    // Find the global max and min.
    let max = Number.NEGATIVE_INFINITY;
    let min = Number.POSITIVE_INFINITY;
    for (let heatMap of heatMaps) {
      if (heatMap.groupMax > max) {
        max = heatMap.groupMax;
      }
      if (heatMap.groupMin < min) {
        min = heatMap.groupMin;
      }
    }
    // Set the global max and min and update the HeatMap.
    for (let heatMap of heatMaps) {
      heatMap.globalMax = max;
      heatMap.globalMin = min;
      heatMap.update();
    }
  }

  private getHeatMapDefault(name: string, rows: number,
                            heatMaps: Map<string, HeatMap>, cache: HeatMap[]) {
    if (!heatMaps.has(name)) {
      let heatMap = new HeatMap(name, this.heatMapHistory, rows);
      heatMaps.set(name, heatMap);
      cache.push(heatMap)
    }
    return heatMaps.get(name);
  }
}
