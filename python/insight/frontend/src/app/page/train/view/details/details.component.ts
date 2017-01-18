import {Component, OnInit, Input, ViewEncapsulation, ViewChild} from '@angular/core';
import {Subscription} from "rxjs";
import {
  TrainerBatchResult,
  Delta
} from "../../../../common/service/api/insight-api-message";
import {TrainerDomain} from "../../../../common/domain/trainer";
import {
  HeatMap,
  HeatMapComponent,
  HeatMapMode,
  GroupColumnSelectionEvent
} from "../../../../common/component/heat-map/heat-map.component";
import {UiFriendlyEnum} from "../../../../common/domain/ui-friendly-enum";
import {PolymerElement} from "@vaadin/angular2-polymer";
import {formatPercent} from "../../../../common/util/parse";

const TABS = [ "Heat Map", "Graphs" ];
const DELTA_ROW_NAME = 'd';
const GRADIENT_ROW_NAME = 'g';
const WEIGHT_ROW_NAME = 'w';
const METRIC_NAMES = [ DELTA_ROW_NAME, GRADIENT_ROW_NAME, WEIGHT_ROW_NAME ];

export class ParameterDetail {
  weightChange: string;

  constructor(public epoch: number, public name: string, public delta: Delta,
              public gradient: number, public weight: number) {
    let deltaValue = this.delta.value;
    this.weightChange = formatPercent(deltaValue / (this.weight - deltaValue));
  }
}

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
  detail: ParameterDetail;
  heatMapHistory: number = 100;
  tabIndex: number = 0;
  heatMapMode: UiFriendlyEnum<HeatMapMode> =
      new UiFriendlyEnum<HeatMapMode>(HeatMapMode);
  visibleMetrics: string[] = Array.from(METRIC_NAMES);
  showParameterNames: boolean = false;
  showParameterDivider: boolean = true;
  useLogScale: boolean = true;

  @ViewChild(HeatMapComponent)
  heatMapComponent: HeatMapComponent;

  private _trainer: TrainerDomain;
  private batchResultSubscription: Subscription;
  private parameterDetails = new Map<string, ParameterDetail>();

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

  onGroupColumnSelection(event: GroupColumnSelectionEvent) {
    let key = this.parameterDetailKey(event.groupName, event.column);
    this.detail = this.parameterDetails.get(key);
  }

  metricDisplayName(metric: string) {
    switch (metric) {
      case DELTA_ROW_NAME:
        return 'Deltas';
      case GRADIENT_ROW_NAME:
        return 'Gradients';
      case WEIGHT_ROW_NAME:
        return 'Weights';
      default:
        throw new Error(`No display name for metric [${metric}]`);
    }
  }

  private updateParameters(batchResult: TrainerBatchResult) {
    let paramSetMaps = batchResult.parameters;
    for (let paramSetMap of paramSetMaps) {
      for (let key in paramSetMap) {
        let paramSet = paramSetMap[key];
        let name = paramSet.name;
        let deltas = [].concat(...paramSet.deltas);
        let gradients = [].concat(...paramSet.gradients);
        let weights = [].concat(...paramSet.values);

        for (let i = 0; i < deltas.length; i++) {
          let groupName = `${name}_${i}`;
          let delta = deltas[i];
          let gradient = gradients[i];
          let weight = weights[i];
          this.heatMap.getGroup(groupName)
              .getRow(DELTA_ROW_NAME)
              .addValue(delta.value);
          this.heatMap.getGroup(groupName)
              .getRow(GRADIENT_ROW_NAME)
              .addValue(gradient);
          this.heatMap.getGroup(groupName)
              .getRow(WEIGHT_ROW_NAME)
              .addValue(weight);
          let epoch = batchResult.batchNumber;
          let detail = new ParameterDetail(epoch, groupName,
                                           delta, gradient, weight);
          this.parameterDetails.set(
              this.parameterDetailKey(detail.name, detail.epoch), detail);
          if (this.detail == null) {
            this.heatMapComponent.selectGroupColumn(groupName, DELTA_ROW_NAME, epoch);
          }
        }
      }
    }
    this.heatMap.update();
  }

  private parameterDetailKey(name: string, epoch: number): string {
    return `${name}:${epoch}`;
  }
}
