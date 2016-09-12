export interface TypeScriptEnum {
  [key: string]: any;
  [key: number]: any;
}

export class UiFriendlyEnum<T> {
  private enumStrings: string[];
  private _displayNames: string[];

  index: number;

  constructor(private tsEnum: TypeScriptEnum) {
    this.enumStrings = Object.keys(this.tsEnum).filter(key => Number.isNaN(Number(key)));
    this._displayNames = this.enumStrings.map(this.toDisplayName);
    this.index = 0;
  }

  get displayNames(): string[] {
    return this._displayNames;
  }

  get displayName(): string {
    return this.displayNames[this.index];
  }

  set displayName(name: string) {
    this.index = this.displayNames.indexOf(name);
  }

  get value(): T {
    return this.tsEnum[this.enumStrings[this.index]] as T;
  }

  set value(type: T) {
    this.index = this.enumStrings.indexOf(this.tsEnum[type as any] as string);
  }

  private toDisplayName(str: string): string {
    let prettyName = str.toLocaleLowerCase();
    prettyName = prettyName.replace(/_/g, " ");
    return prettyName[0].toLocaleUpperCase() + prettyName.substr(1);
  }
}
