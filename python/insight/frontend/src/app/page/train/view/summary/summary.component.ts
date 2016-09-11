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

export class GoogleChart<T> {
  private static loadedPromise: Promise<any> = null;
  private chart: google.visualization.ChartWrapper;

  constructor(private container: ElementRef, private chartType: string) {
    if (GoogleChart.loadedPromise == null) {
      google.charts.load('current', {'packages' : [ 'corechart', 'gauge' ]});
      GoogleChart.loadedPromise = new Promise(function(resolve, reject) {
        google.charts.setOnLoadCallback(resolve);
      });
    }
    GoogleChart.loadedPromise.then(() => {
      this.chart = new google.visualization.ChartWrapper({chartType});
    });
  }

  draw(data: google.visualization.DataTable, options: T) {
    this.chart.setDataTable(data);
    this.chart.setOptions(options);
    this.chart.draw(this.container.nativeElement)
  }
}

const BATCH_CHART_OPTIONS: google.visualization.SteppedAreaChartOptions = {
  title : 'Batch Performance',
  legend : {position : 'right'}
};

const VALIDATION_CHART_OPTIONS: google.visualization.LineChartOptions = {
  title : 'Validation Results',
  curveType : 'function',
  legend : {position : 'right'}
};

@Component({
  moduleId : module.id,
  selector : 'app-summary',
  templateUrl : 'summary.component.html',
  styleUrls : [ 'summary.component.css' ],
  encapsulation : ViewEncapsulation.Native
})
export class SummaryComponent implements OnInit, AfterViewInit {
  private _trainer: TrainerDomain = null;
  private batchResultSubscription: Subscription = null;
  private batchChart: GoogleChart<google.visualization.ScatterChartOptions>;
  private validationChart: GoogleChart<google.visualization.LineChartOptions>;
  private lastValidationResult: TrainerValidationResult;

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

  private onBatchResult(result: TrainerBatchResult): void {
    this.updateBatchChart(result);
    this.trainer.validate().subscribe((result) =>
                                          this.updateValidationChart(result));
  }

  private updateBatchChart(result: TrainerBatchResult) {
    let dataTable = new google.visualization.DataTable();
    dataTable.addColumn('number', 'Input');
    dataTable.addColumn('number', 'Expected');
    dataTable.addColumn('number', 'Actual');

    let data = [];
    for (let i = 0; i < result.inputs.length; i++) {
      data.push(
          [ result.inputs[i][0], result.expected[i][0], result.actual[i][0] ])
    }
    data.sort((a: number[], b: number[]) => a[0] - b[0]);
    dataTable.addRows(data);
    this.batchChart.draw(dataTable, BATCH_CHART_OPTIONS);
  }

  private updateValidationChart(result: TrainerValidationResult) {
    let dataTable = new google.visualization.DataTable();
    dataTable.addColumn('number', 'Input');
    dataTable.addColumn('number', 'Expected');
    dataTable.addColumn('number', 'Current');
    if (this.lastValidationResult != null) {
      dataTable.addColumn('number', 'Prior');
    }

    let data = [];
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
    this.validationChart.draw(dataTable, VALIDATION_CHART_OPTIONS);
    this.lastValidationResult = result;
  }
}
