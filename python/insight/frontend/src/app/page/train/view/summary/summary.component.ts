import {
  Component, OnInit, ViewEncapsulation, Input, ElementRef, ViewChild, AfterViewInit
} from '@angular/core';
import {TrainerDomain} from "../../../../common/domain/trainer";
import {TrainerBatchResult} from "../../../../common/service/api/insight-api-message";
import {Subscription} from "rxjs";

export class GoogleChart<T> {
  private static loadedPromise: Promise<any> = null;
  private chart: google.visualization.ChartWrapper;

  constructor(private container: ElementRef, private chartType: string) {
    if (GoogleChart.loadedPromise == null) {
      google.charts.load('current', {'packages':['corechart', 'gauge']});
      GoogleChart.loadedPromise = new Promise(function (resolve, reject) {
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
  private currentBatch: google.visualization.DataTable;
  private priorBatch: google.visualization.DataTable;
  private currentChart: GoogleChart<google.visualization.LineChartOptions>;
  private priorChart: GoogleChart<google.visualization.LineChartOptions>;
  private chartOptions = {
    title : 'Batch Results',
    curveType : 'function',
    legend : {position : 'right'}
  };



  @ViewChild('priorChart')
  priorChartElement: ElementRef;

  @ViewChild('currentChart')
  currentChartElement: ElementRef;

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
    this.currentChart = new GoogleChart(this.currentChartElement, "LineChart");
    this.priorChart = new GoogleChart(this.priorChartElement, "LineChart");
  }

  private onBatchResult(result: TrainerBatchResult): void {
    this.priorBatch = this.currentBatch;
    this.currentBatch = new google.visualization.DataTable();

    this.currentBatch.addColumn('number', 'Input');
    this.currentBatch.addColumn('number', 'Expected');
    this.currentBatch.addColumn('number', 'Actual');
    let data = [];
    for (let i = 0; i < result.inputs.length; i++) {
      data.push([result.inputs[i][0], result.expected[i][0], result.actual[i][0]])
    }
    data.sort((a: [number, number, number], b: [number, number, number]) => a[0] - b[0]);
    this.currentBatch.addRows(data);
    this.currentChart.draw(this.currentBatch, this.chartOptions);
    if (this.priorBatch != null) {
      this.priorChart.draw(this.priorBatch, this.chartOptions);
    }
  }
}
