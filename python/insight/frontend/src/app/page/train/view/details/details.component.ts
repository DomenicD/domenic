import {Component, OnInit, Input, ViewEncapsulation} from '@angular/core';
import {Subscription} from "rxjs";
import {TrainerBatchResult} from "../../../../common/service/api/insight-api-message";
import {TrainerDomain} from "../../../../common/domain/trainer";
import {
  HeatMap,
  HeatMapComponent,
  HeatMapMode
} from "../../../../common/component/heat-map/heat-map.component";
import {UiFriendlyEnum} from "../../../../common/domain/ui-friendly-enum";
import {PolymerElement} from "@vaadin/angular2-polymer";

const TABS = [ "Heat Map", "Graphs" ];
const DELTA_ROW_NAME = 'd';
const GRADIENT_ROW_NAME = 'g';
const WEIGHT_ROW_NAME = 'w';
const METRIC_NAMES = [ DELTA_ROW_NAME, GRADIENT_ROW_NAME, WEIGHT_ROW_NAME ];

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
    PolymerElement('paper-checkbox'),
    PolymerElement('paper-toggle-button'),
  ],
  encapsulation : ViewEncapsulation.Native
})
export class DetailsComponent implements OnInit {

  heatMap: HeatMap = new HeatMap();
  heatMapHistory: number = 100;
  tabIndex: number = 0;
  heatMapMode: UiFriendlyEnum<HeatMapMode> =
      new UiFriendlyEnum<HeatMapMode>(HeatMapMode);
  visibleMetrics: string[] = Array.from(METRIC_NAMES);
  showParameterNames: boolean = true;
  showParameterDivider: boolean = false;
  useLogScale: boolean = false;

  private _trainer: TrainerDomain;
  private batchResultSubscription: Subscription;

  constructor() {}

  ngOnInit() {}

  get metricNames(): string[] { return METRIC_NAMES; }

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

  isMetricVisible(metric: string): boolean {
    return this.visibleMetrics.indexOf(metric) > -1;
  }

  onMetricVisibilityChanged(metric: string, isVisible: boolean) {
    let index = this.visibleMetrics.indexOf(metric);
    let metricVisible = this.isMetricVisible(metric);
    if (isVisible && !metricVisible) {
      this.visibleMetrics.push(metric);
    } else if (!isVisible && metricVisible) {
      this.visibleMetrics.splice(index, 1);
    }
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

        for (let i = 0; i < deltas.length; i++) {
          let groupName = `${name}_${i}`;
          this.heatMap.getGroup(groupName).getRow(DELTA_ROW_NAME).addValue(deltas[i]);
          this.heatMap.getGroup(groupName).getRow(GRADIENT_ROW_NAME).addValue(gradients[i]);
          this.heatMap.getGroup(groupName).getRow(WEIGHT_ROW_NAME).addValue(weights[i]);
        }
      }
    }
    this.heatMap.update();
  }
}
