/**
 * 页面唯一的可变状态与最小的订阅机制。
 *
 * 分层约定（单向依赖，避免 ES 模块循环引用）：
 *   store → api / util → model → shell → sections → app.js
 * 数据刷新后由 shell 广播 `overview`，各页面自己决定要不要重绘，
 * 不再由 shell 逐个调用各页的 render 函数。
 */
export const state = {
  meta: null,
  rules: [],
  bindings: [],
  status: [],
  section: 'rules',
  editingRuleId: null,
  editingRule: null,
  ruleDirty: false,
  rulePage: 1,
  selectedPlatform: '',
  editingBindingId: null,
  editingBinding: null,
  bindingDirty: false,
  sessions: [],
  /** 「更多设置」里正在编辑的自定义决策模型（保存前只存在于内存）。 */
  decisionModels: [],
  /** 卡片序号 → 测试结果（loading / ok / error），不随保存提交。 */
  decisionTestResults: {},
  /** 启停请求；启动在途时仍允许发起取消，旧响应不能覆盖新操作。 */
  busyBindings: new Map(),
  bindingToggleVersions: new Map(),
  overviewRequest: 0,
  statusRequest: 0,
  /** 缓冲区弹窗正在查看的绑定 id 与目标会话；bindingId 为 null 表示弹窗关闭。 */
  bufferBindingId: null,
  bufferUmo: '',
  /** 缓冲区快照（loading / error / data），只读、不随运行状态轮询刷新。 */
  bufferView: null,
  /** 手动触发在途的目标会话（`binding_id|umo`）：轮询重绘时保持按钮禁用。 */
  busyForcedTargets: new Set(),
  reapplying: false,
};

const listeners = {
  /** 概览数据（rules / bindings / settings）已更新，各页面重绘自己。 */
  overview: [],
  /** 一次保存流程走完（含刷新），需要额外拉取数据的页面在这里补。 */
  afterSave: [],
  /** 侧栏切换页面。 */
  section: [],
};

export function on(event, handler) {
  listeners[event].push(handler);
}

/** 依次等待所有订阅者。返回 Promise，调用方按需 await。 */
export function emit(event, payload) {
  return Promise.all(listeners[event].map((handler) => handler(payload)));
}
