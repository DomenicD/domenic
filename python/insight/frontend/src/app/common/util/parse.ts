export function toNumbers(list: Array<number | string>): number[] {
  return list.map((str: number | string) => toNumber(str));
}

export function toNumber(value: string | number): number {
  return Number(value);
}

export function formatNumber(value: number) {
  return Math.round(value).toLocaleString();
}

export function formatPercent(value: number) {
  return formatNumber(value * 100) + "%";
}
