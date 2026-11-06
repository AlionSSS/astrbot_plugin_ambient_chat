/** 页面外壳：顶部横幅、侧栏导航、保存后的收尾流程。 */
import { apiGet, apiPost } from './api.js';
import { emit, state } from './store.js';
import { toast } from './ui.js';
import { $, clone } from './util.js';

/**
 * 拉取概览数据并广播 `overview`。
 * 各页面（规则 / 绑定 / 设置）自己订阅重绘，shell 不直接调用它们的 render。
 */
export async function refresh() {
  const request = ++state.overviewRequest;
  const data = await apiGet('overview');
  if (request !== state.overviewRequest) return;
  state.meta = data;
  state.rules = data.rules || [];
  state.bindings = data.bindings || [];

  if (state.editingRuleId) {
    const fresh = state.rules.find((rule) => rule.id === state.editingRuleId);
    if (!fresh) {
      state.editingRuleId = null;
      state.editingRule = null;
    } else if (!state.ruleDirty) {
      state.editingRule = clone(fresh);
    }
  }

  renderBanner();
  renderNav();
  await emit('overview');
}

export function renderBanner() {
  renderApplication();
  const banner = $('#banner');
  const error = state.meta?.load_error || '';
  if (error) {
    banner.textContent = `配置文件异常：${error}`;
    banner.classList.remove('notice');
    banner.classList.remove('hidden');
    return;
  }
  if (state.meta?.enabled === false) {
    banner.textContent =
      '「启用自主发言」已关闭：调度已暂停，规则与绑定仍可编辑；恢复请在 AstrBot 插件配置中打开该开关。';
    banner.classList.add('notice');
    banner.classList.remove('hidden');
    return;
  }
  banner.classList.add('hidden');
  banner.classList.remove('notice');
}

export function renderApplication() {
  const application = state.meta?.application;
  const banner = $('#application-banner');
  const failed = application?.state === 'failed';
  const applying = application?.state === 'applying';
  banner.classList.toggle('hidden', !failed && !applying);
  banner.classList.toggle('notice', !failed);
  $('#application-message').textContent = failed
    ? '配置已保存，运行应用失败；部分更改可能已生效，请重新应用。'
    : '配置已保存，正在应用到运行中…';
  const button = $('#btn-apply-configuration');
  button.classList.toggle('hidden', !failed);
  button.disabled = state.reapplying;
  button.textContent = state.reapplying ? '应用中…' : '重新应用';
}

/**
 * 保存接口统一的后半程：刷新概览 → 让需要额外数据的页面补拉 → 按应用状态提示。
 * `isCurrent` 用于丢弃过期响应（启停请求会并发）。
 */
export async function finishSaved(data, message, isCurrent = () => true) {
  if (!isCurrent()) return;
  if (state.meta && data?.application) state.meta.application = data.application;
  renderBanner();
  renderRunBadge();
  try {
    await refresh();
    if (!isCurrent()) return;
    await emit('afterSave');
    if (!isCurrent()) return;
  } catch (error) {
    if (isCurrent()) toast(`配置已保存，刷新失败：${error.message || '请点击刷新'}`, true);
    return;
  }
  const application = state.meta?.application || data?.application;
  if (application?.state === 'failed') {
    toast('配置已保存，运行应用失败，请点击「重新应用」', true);
  } else if (application?.state === 'applying') {
    toast('配置已保存，运行应用尚未完成');
  } else {
    const text = typeof message === 'function' ? message() : message;
    if (text) toast(text);
  }
}

export async function applyConfiguration() {
  if (state.reapplying) return;
  state.reapplying = true;
  renderApplication();
  try {
    const data = await apiPost('configuration/apply', {});
    await finishSaved(data, '已保存配置已重新应用');
  } catch (error) {
    toast(error.message || '重新应用失败', true);
  } finally {
    state.reapplying = false;
    renderApplication();
  }
}

export function renderNav() {
  renderRunBadge();
  $('#nav-count-rules').textContent = String(state.rules.length);
  $('#nav-count-bindings').textContent = String(state.bindings.length);
  const running = state.bindings.filter((item) => item.enabled).length;
  $('#nav-count-status').textContent = String(running);
}

/** 侧栏状态灯：一眼看出「在不在干活 / 配置有没有坏」，详情交给 tooltip。 */
export function renderRunBadge() {
  const badge = $('#run-badge');
  if (!badge) return;
  const broken = Boolean(state.meta?.load_error) || state.meta?.application?.state === 'failed';
  const enabled = state.meta?.enabled !== false;
  badge.classList.toggle('ok', !broken && enabled);
  badge.classList.toggle('off', !broken && !enabled);
  badge.classList.toggle('danger', broken);
  if (broken) {
    badge.textContent = state.meta?.load_error ? '配置异常' : '应用失败';
    badge.title = state.meta?.load_error
      ? '配置文件加载失败，调度已停止；详情见「更多设置 → 运行信息」。'
      : '配置已保存，运行应用失败；请点击页面上方的「重新应用」。';
    return;
  }
  if (enabled) {
    badge.textContent = '调度中';
    badge.title = '「启用自主发言」已打开，绑定按各自的入口触发调度。';
    return;
  }
  badge.textContent = '已暂停';
  badge.title = '「启用自主发言」已关闭，调度暂停；规则与绑定仍可编辑。';
}

/** 切页：只切换可见性，页内行为交给各自的 `section` 订阅者。 */
export function showSection(name) {
  state.section = name;
  for (const button of document.querySelectorAll('.nav-item')) {
    button.classList.toggle('active', button.dataset.section === name);
  }
  for (const section of document.querySelectorAll('.section')) {
    section.classList.toggle('hidden', section.id !== `section-${name}`);
  }
  emit('section', name);
}
