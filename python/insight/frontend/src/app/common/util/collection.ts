export function getDefault<K, V>(m: Map<K, V>, key: K, defaultValue: V): V {
    if (!m.has(key)) {
      m.set(key, defaultValue);
    }
    return m.get(key);
  }
