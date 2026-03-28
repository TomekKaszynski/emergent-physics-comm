/**
 * WMCP JavaScript Client
 * Connects to a WMCP FastAPI server.
 *
 * Usage:
 *   const wmcp = require('wmcp-client');
 *   const client = new wmcp.WMCPClient('http://localhost:8000');
 *   const health = await client.health();
 *   const result = await client.communicate();
 */

class WMCPClient {
  /**
   * @param {string} baseUrl - WMCP server URL (e.g., 'http://localhost:8000')
   */
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async _fetch(path, options = {}) {
    const url = `${this.baseUrl}${path}`;
    const res = await fetch(url, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) {
      throw new Error(`WMCP API error: ${res.status} ${res.statusText}`);
    }
    return res.json();
  }

  /** Get server health status. */
  async health() {
    return this._fetch('/health');
  }

  /** Run a communication round (synthetic data). */
  async communicate() {
    return this._fetch('/communicate', { method: 'POST' });
  }

  /** Run compliance validation. */
  async validate() {
    return this._fetch('/validate', { method: 'POST' });
  }

  /** Get Prometheus metrics (text format). */
  async metrics() {
    const res = await fetch(`${this.baseUrl}/metrics`);
    return res.text();
  }

  /** Get dashboard HTML. */
  async dashboard() {
    const res = await fetch(`${this.baseUrl}/dashboard`);
    return res.text();
  }
}

module.exports = { WMCPClient };
