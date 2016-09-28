import {Component, OnInit, ViewEncapsulation, Input} from '@angular/core';
import {getDefault} from "../../util/collection";

export enum HeatMapMode {
  LOCAL,
  GLOBAL
}

const DEFAULT_HISTORY = 100;
const DEFAULT_MODE = HeatMapMode.LOCAL;

export class HeatMapCell {
  relativeValue: number = 0;

  constructor(public actualValue: number) {}
}

export class HeatMapRow {
  history: number = DEFAULT_HISTORY;
  globalMax: number = 0;
  globalMin: number = 0;
  localMax: number = 0;
  localMin: number = 0;
  mode: HeatMapMode = DEFAULT_MODE;
  visibleCells: HeatMapCell[] = [];

  private cells: HeatMapCell[] = [];

  constructor(public name: string) {}

  addValue(value: number) {
    this.cells.unshift(new HeatMapCell(value));
    if (this.cells.length > this.history) {
      this.updateLocalMinMax(value,
        this.cells[this.history].actualValue);
    } else {
      this.updateLocalMinMax(value);
    }
  }

  update() {
    this.visibleCells = this.cells.slice(0, this.history);
    for (let cell of this.visibleCells) {
      cell.relativeValue = this.scaleValue(cell.actualValue);
    }
  }

  private updateLocalMinMax(value: number, removed?: number) {
    if (removed == this.localMax) {
      this.localMax = Math.max(...this.cells.map(c => c.actualValue));
    } else if (value > this.localMax) {
      this.localMax = value;
    }

    if (removed == this.localMin) {
      this.localMin = Math.min(...this.cells.map(c => c.actualValue));
    } else if (value < this.localMin) {
      this.localMin = value;
    }
  }

  private scaleValue(value: number): number {
    let max = 0;
    let min = 0;
    switch (this.mode) {
    case HeatMapMode.LOCAL:
      max = this.localMax;
      min = this.localMin;
      break;
    case HeatMapMode.GLOBAL:
      max = this.globalMax;
      min = this.globalMin;
      break;
    default:
      throw new Error(`HeatMapMode ${HeatMapMode[this.mode]} not implemented`);
    }
    let absMax = Math.max(Math.abs(max), Math.abs(min));
    let scaled = absMax > 0 ? value / absMax : 0;
    if (Math.abs(scaled) > 1) {
      console.error("Scaled should be clamped between 1 and -1");
    }
    return scaled;
  }
}

export class HeatMapGroup {
  private _history: number = DEFAULT_HISTORY;
  private _mode: HeatMapMode = DEFAULT_MODE;

  constructor(public name: string, public rows: HeatMapRow[] = []) { }

  get history(): number { return this._history; }

  set history(value: number) {
    for (let row of this.rows) {
      row.history = value;
    }
    this._history = value;
  }

  get mode(): HeatMapMode { return this._mode; }

  set mode(value: HeatMapMode) {
    for (let row of this.rows) {
      row.mode = value;
    }
    this._mode = value;
  }

  getRow(rowName: string): HeatMapRow {
    let row = this.rows.find(r => r.name === rowName);
    if (row == null) {
      row = new HeatMapRow(rowName);
      this.rows.push(row);
    }
    return row;
  }

  setGlobalExtrema(rowKey: string, min: number, max: number) {
    this.getRow(rowKey).globalMin = min;
    this.getRow(rowKey).globalMax = max;
  }

  update() {
    for (let row of this.rows) {
      row.update();
    }
  }
}

export class HeatMap {
  private _history: number = DEFAULT_HISTORY;
  private _mode: HeatMapMode = DEFAULT_MODE;

  constructor(public groups: HeatMapGroup[] = []) { }

  get history(): number { return this._history; }

  set history(value: number) {
    for (let group of this.groups) {
      group.history = value;
    }
    this._history = value;
  }

  get mode(): HeatMapMode { return this._mode; }

  set mode(value: HeatMapMode) {
    for (let group of this.groups) {
      group.mode = value;
    }
    this._mode = value;
  }

  getGroup(groupName: string): HeatMapGroup {
    let group = this.groups.find(r => r.name === groupName);
    if (group == null) {
      group = new HeatMapGroup(groupName);
      this.groups.push(group);
    }
    return group;
  }

  update() {
    let maxes = new Map<string, number>();
    let mins = new Map<string, number>();

    // Find the global min and max for each row type.
    for (let group of this.groups) {
      for (let row of group.rows) {
        let rowName = row.name;
        let max = getDefault(maxes, rowName, Number.NEGATIVE_INFINITY);
        let min = getDefault(mins, rowName, Number.POSITIVE_INFINITY);
        if (row.localMax > max) {
          maxes.set(rowName, row.localMax);
        }
        if (row.localMin < min) {
          mins.set(rowName, row.localMin);
        }
      }
    }

    // Set the global min and max for each row type.
    for (let group of this.groups) {
      for (let row of group.rows) {
        row.globalMax = maxes.get(row.name);
        row.globalMin = mins.get(row.name);
      }
      // Have the group perform its update now that all its rows have been updated.
      group.update();
    }
  }
}

@Component({
  moduleId : module.id,
  selector : 'app-heat-map',
  templateUrl : 'heat-map.component.html',
  styleUrls : [ 'heat-map.component.css' ],
  encapsulation : ViewEncapsulation.Native
})
export class HeatMapComponent implements OnInit {
  private _history: number = DEFAULT_HISTORY;
  private _mode: HeatMapMode = DEFAULT_MODE;

  @Input() heatMap: HeatMap;
  @Input() showGroupName: boolean = true;
  @Input() visibleRows: string[] = [];

  constructor() {}

  @Input()
  get history(): number {
    return this._history;
  }

  set history(value: number) {
    if (this.heatMap != null && this.history != value) {
      this.heatMap.history = value;
      this.heatMap.update();
      this._history = value;
    }
  }

  @Input()
  get mode(): HeatMapMode {
    return this._mode;
  }

  set mode(value: HeatMapMode) {
    if (this.heatMap != null && this.mode != value) {
      this.heatMap.mode = value;
      this.heatMap.update();
      this._mode = value;
    }
  }

  ngOnInit() {}

  showRow(name: string): boolean {
    return this.visibleRows.indexOf(name) > -1;
  }

  getColor(value: number) {
    let percent = ((1 - Math.abs(value)) * 100).toFixed(0) + "%";
    return value > 0 ? `rgb(100%, ${percent}, ${percent})`
                     : `rgb(${percent}, ${percent}, 100%)`;
  }
}
