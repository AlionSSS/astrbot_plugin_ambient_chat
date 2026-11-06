/**
 * 视图入口：启动流程 + 全局事件委托。
 *
 * 分层（单向依赖）：
 *   store（状态）→ api / util → model → shell（外壳）→ sections（各页）→ 本文件
 * 各页的按钮通过导出的 `actions` 表登记，这里只做一次派发，
 * 不再把所有按钮的 if 分支堆在一个回调里。
 */
import { ready } from './js/api.js';
import { maybeShowAstrbotConfigAdvice } from './js/first-run.js';
import { renderBindings } from './js/sections/bindings.js';
import {
  actions as bindingModalActions,
  openBindingModal,
  requestCloseBindingModal,
} from './js/sections/binding-modal.js';
import { actions as bufferActions, closeBufferModal } from './js/sections/buffer.js';
import { actions as ruleActions, openRuleModal, requestCloseRuleModal } from './js/sections/rules.js';
import { actions as settingsActions } from './js/sections/settings.js';
import { actions as statusActions, loadStatus } from './js/sections/status.js';
import { applyConfiguration, refresh, showSection } from './js/shell.js';
import { state } from './js/store.js';
import { closePinnedHelp, setHelpOpen, toast } from './js/ui.js';

/** 按钮动作表：`id` 对应元素 id，`data:xxx` 对应 `data-xxx` 属性。 */
const idActions = new Map();
const dataActions = new Map();

for (const table of [
  ruleActions,
  bindingModalActions,
  statusActions,
  bufferActions,
  settingsActions,
]) {
  for (const [key, run] of Object.entries(table)) {
    if (key.startsWith('data:')) dataActions.set(key.slice(5), run);
    else idActions.set(key, run);
  }
}

// 入口自己负责的动作：需要同时驱动外壳与运行状态页。
idActions.set('btn-refresh', async () => {
  await refresh();
  if (state.section === 'status') await loadStatus();
  toast('已刷新');
});
idActions.set('btn-apply-configuration', () => applyConfiguration());
// 侧栏切页：任何带 `data-section` 的按钮都算导航。
dataActions.set('section', (name) => showSection(name));

/** 找出按钮对应的动作；命中 `data-*` 时把属性值与按钮一起传进去。 */
function resolveAction(button) {
  const byId = button.id ? idActions.get(button.id) : null;
  if (byId) return () => byId(button);
  for (const [key, run] of dataActions) {
    if (button.dataset[key] !== undefined) {
      return () => run(button.dataset[key], button);
    }
  }
  return null;
}

// Esc 关闭弹窗；确认框自己监听 Esc，同时打开时让它先处理。
document.addEventListener('keydown', (event) => {
  if (event.key !== 'Escape') return;
  // 说明卡片最轻，先关它，不连带关闭弹窗。
  if (closePinnedHelp()) return;
  if (!document.querySelector('#dialog').classList.contains('hidden')) return;
  if (!document.querySelector('#buffer-modal').classList.contains('hidden')) {
    closeBufferModal();
    return;
  }
  if (!document.querySelector('#rule-modal').classList.contains('hidden')) {
    requestCloseRuleModal();
    return;
  }
  if (document.querySelector('#binding-modal').classList.contains('hidden')) return;
  requestCloseBindingModal();
});

document.addEventListener('click', async (event) => {
  const button = event.target.closest('button');
  const ruleItem = event.target.closest('[data-rule-id]');
  const botItem = event.target.closest('[data-bot-id]');
  const bindingItem = event.target.closest('[data-binding-id]');

  // 点空白处收起「钉住」的说明卡片；悬停打开的那些本来就跟着鼠标走。
  if (!event.target.closest('.section-help')) closePinnedHelp();

  if (event.target.id === 'rule-modal') {
    await requestCloseRuleModal();
    return;
  }
  if (event.target.id === 'binding-modal') {
    await requestCloseBindingModal();
    return;
  }
  if (event.target.id === 'buffer-modal') {
    closeBufferModal();
    return;
  }
  // 行本身可点开编辑；行内按钮走下面的动作表。
  if (ruleItem && !button) {
    openRuleModal(ruleItem.dataset.ruleId);
    return;
  }
  if (botItem && !button) {
    state.selectedPlatform = botItem.dataset.botId;
    renderBindings();
    return;
  }
  if (bindingItem && !button) {
    openBindingModal(bindingItem.dataset.bindingId);
    return;
  }
  if (!button) return;

  if (button.classList.contains('help-icon')) {
    const help = button.closest('.section-help');
    if (help) setHelpOpen(help, !help.classList.contains('is-open'));
    return;
  }

  const action = resolveAction(button);
  if (action) await action();
});

async function boot() {
  await ready();
  await refresh();
  showSection('rules');
  // 首次进入控制台的一次性提示：不等待用户关闭，弹窗自己处理交互与标记写入。
  void maybeShowAstrbotConfigAdvice();
}

boot().catch((error) => {
  toast(`初始化失败：${error.message || error}`, true);
});
