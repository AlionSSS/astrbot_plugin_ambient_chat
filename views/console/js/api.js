/**
 * 与 AstrBot 的桥接。
 * `bridge` 由 dashboard 注入的 bridge-sdk.js 提供（视图运行在受限 iframe 里，
 * 不能直接用 fetch 调后端接口）。
 */
const bridge = window.AstrBotPluginView;

/** 等桥接握手完成；页面初始化必须先 await 它。 */
export function ready() {
  return bridge.ready();
}

/** 后端统一返回 `{ status, data }`，这里只在需要时拆一层。 */
function unwrap(payload) {
  if (payload && typeof payload === 'object' && 'status' in payload && 'data' in payload) {
    return payload.data;
  }
  return payload;
}

export async function apiGet(path, params) {
  return unwrap(await bridge.apiGet(path, params));
}

export async function apiPost(path, body) {
  return unwrap(await bridge.apiPost(path, body));
}
