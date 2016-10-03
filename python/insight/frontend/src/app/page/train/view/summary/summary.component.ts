import {
  Component,
  OnInit,
  ViewEncapsulation,
  Input,
  ElementRef,
  ViewChild,
  AfterViewInit
} from '@angular/core';
import {TrainerDomain} from "../../../../common/domain/trainer";
import {
  TrainerBatchResult,
  TrainerValidationResult
} from "../../../../common/service/api/insight-api-message";
import {Subscription} from "rxjs";
import {PolymerElement} from "@vaadin/angular2-polymer";
import {formatNumber, formatPercent} from "../../../../common/util/parse";

const MAX_HISTORY = 50;

export class GoogleChart<T> {
  private chart: google.visualization.ChartWrapper;

  constructor(private container: ElementRef, private chartType: string) {
    this.chart = new google.visualization.ChartWrapper({chartType});
  }

  draw(data: google.visualization.DataTable, options: T) {
    this.chart.setDataTable(data);
    this.chart.setOptions(options);
    this.chart.draw(this.container.nativeElement)
  }
}

export class TrainingSummary {
  epoch: string;
  batchError: string;
  batchDelta: string;
  validationError: string;
  validationDelta: string;
  constructor(epoch: number, batchError: number, lastBatchError: number,
              validationError: number, lastValidationError: number) {
    this.epoch = formatNumber(epoch);
    this.batchError = formatNumber(batchError);
    this.batchDelta = formatPercent(
        (batchError - lastBatchError) / lastBatchError);
    this.validationError = formatNumber(validationError);
    this.validationDelta = formatPercent(
        (validationError - lastValidationError) / lastValidationError);
  }
}

@Component({
  moduleId : module.id,
  selector : 'app-summary',
  templateUrl : 'summary.component.html',
  styleUrls : [ 'summary.component.css' ],
  encapsulation : ViewEncapsulation.Native,
  directives : [ PolymerElement('vaadin-grid') ]
})
export class SummaryComponent implements OnInit,
    AfterViewInit {
  private _trainer: TrainerDomain;
  private batchResultSubscription: Subscription;
  private batchChart: GoogleChart<google.visualization.ScatterChartOptions>;
  private validationChart: GoogleChart<google.visualization.LineChartOptions>;
  private lastValidationResult: TrainerValidationResult;

  validationSummaries: TrainingSummary[] = [];
  batchError: number = 0;
  validationError: number = 0;

  @ViewChild('batchChart') batchChartContainer: ElementRef;

  @ViewChild('validationChart') validationChartContainer: ElementRef;

  constructor() {}

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
        (batchResult: TrainerBatchResult) => this.onBatchResult(batchResult));
  }

  ngOnInit() {}

  ngAfterViewInit(): void {
    this.batchChart = new GoogleChart(this.batchChartContainer, "ScatterChart");
    this.validationChart =
        new GoogleChart(this.validationChartContainer, "LineChart");
  }

  private onBatchResult(batchResult: TrainerBatchResult): void {
    this.updateBatchChart(batchResult);
    this.trainer.validate().subscribe(
        (validationResult) =>
            this.onValidationResult(batchResult, validationResult));
  }

  private onValidationResult(batchResult: TrainerBatchResult,
                             validationResult: TrainerValidationResult) {
    this.validationSummaries.unshift(
        new TrainingSummary(batchResult.batchNumber, batchResult.avgError,
          this.batchError, validationResult.error, this.validationError));
    if (this.validationSummaries.length > MAX_HISTORY) {
      this.validationSummaries.pop();
    }

    this.batchError = batchResult.avgError;
    this.validationError = validationResult.error;
    this.updateValidationChart(validationResult);
  }

  private updateBatchChart(result: TrainerBatchResult) {
    let dataTable = new google.visualization.DataTable();
    dataTable.addColumn('number', 'Input');
    dataTable.addColumn('number', 'Expected');
    dataTable.addColumn('number', 'Actual');

    let data: number[][] = [];
    for (let i = 0; i < result.inputs.length; i++) {
      data.push(
          [ result.inputs[i][0], result.expected[i][0], result.actual[i][0] ])
    }
    data.sort((a: number[], b: number[]) => a[0] - b[0]);
    dataTable.addRows(data);
    this.batchChart.draw(dataTable, this.batchChartOptions(result));
  }

  private updateValidationChart(result: TrainerValidationResult) {
    let dataTable = new google.visualization.DataTable();
    dataTable.addColumn('number', 'Input');
    dataTable.addColumn('number', 'Expected');
    dataTable.addColumn('number', 'Current');
    if (this.lastValidationResult != null) {
      dataTable.addColumn('number', 'Prior');
    }

    let data: number[][] = [];
    for (let i = 0; i < result.inputs.length; i++) {
      var rows =
          [ result.inputs[i][0], result.expected[i][0], result.actual[i][0] ];
      if (this.lastValidationResult != null) {
        rows.push(this.lastValidationResult.actual[i][0]);
      }
      data.push(rows)
    }
    data.sort((a: number[], b: number[]) => a[0] - b[0]);
    dataTable.addRows(data);
    this.validationChart.draw(dataTable, this.validationChartOptions(result));
    this.lastValidationResult = result;
  }

  private batchChartOptions(result: TrainerBatchResult):
      google.visualization.ScatterChartOptions {
    return {title : `Batch`, legend : {position : 'bottom'}};
  }

  private validationChartOptions(result: TrainerValidationResult):
      google.visualization.ScatterChartOptions {
    return {
      title : `Validation`,
      curveType : 'function',
      legend : {position : 'bottom'}
    };
  }
}
