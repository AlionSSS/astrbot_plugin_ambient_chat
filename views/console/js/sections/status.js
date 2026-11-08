/** 运行管理页：轮询运行状态，提供启停与手动触发。 */
import { apiGet, apiPost } from '../api.js';
import { platformParts, selectionText, targetParts, triggerExpired } from '../model.js';
import { finishSaved, renderBanner, renderNav } from '../shell.js';
import { on, state } from '../store.js';
import { askConfirm, toast } from '../ui.js';
import { $, escapeHtml, formatCountdown, formatMoment, formatStamp } from '../util.js';

let statusTimer = null;
/** 上一次渲染的指纹：轮询回来只有倒计时变化时不重绘，避免打断 hover 与焦点。 */
let lastFingerprint = '';
/** 停止态卡片默认折叠会话明细；展开状态记在内存里，轮询重绘后保持。 */
const expandedTargets = new Set();
/** 当前筛选：all / running / stopped / problem。 */
let statusFilter = 'all';

const PHASES = {
  queued: '等待并发名额',
  judging: '判断中',
  generating: '生成中',
  pacing: '等待发送时机',
  sending: '发送中',
  recording: '记录中',
};
/** 判定失败与「结果未知 / 历史记录失败 / 部分发送」要用不同颜色区分，不能都读成普通跳过。 */
const DANGER_STATES = new Set(['decision_failed']);
const WARN_STATES = new Set(['unknown', 'history_failed', 'partial']);
const FILTERS = [
  { key: 'all', label: '全部' },
  { key: 'running', label: '运行中' },
  { key: 'stopped', label: '已停止' },
  { key: 'problem', label: '仅异常' },
];

function startStatusPolling() {
  stopStatusPolling();
  statusTimer = setInterval(() => {
    loadStatus().catch(() => {});
  }, 10000);
}

function stopStatusPolling() {
  if (statusTimer) clearInterval(statusTimer);
  statusTimer = null;
}

export async function loadStatus() {
  const request = ++state.statusRequest;
  const data = await apiGet('status');
  if (request !== state.statusRequest) return;
  state.status = data.instances || [];
  if (state.meta) {
    state.meta.application = data.application;
    state.meta.enabled = data.enabled;
  }
  for (const binding of state.bindings) {
    const status = state.status.find((item) => item.binding_id === binding.id);
    if (status) Object.assign(binding, { enabled: status.enabled, lifecycle: status.lifecycle });
  }
  renderBanner();
  renderNav();
  renderStatus();
}

/** 绑定的运行阶段：请求在途的启停优先于后端快照，避免按钮文案来回跳。 */
function lifecycleOf(item) {
  const request = state.busyBindings.get(String(item.binding_id));
  const stopping = item.lifecycle === 'stopping' || request?.enabled === false;
  const starting = !stopping && (item.lifecycle === 'starting' || request?.enabled === true);
  const running = !stopping && !starting && (item.ready ?? Boolean(item.running));
  const enabled = Boolean(item.enabled);
  const label = stopping
    ? '停止收尾中'
    : starting
      ? '启动预热中'
      : running
        ? '运行中'
        : enabled
          ? '等待应用'
          : '已停止';
  const tone = stopping || starting ? 'pending' : running ? 'on' : enabled ? 'warn' : 'off';
  return { request, stopping, starting, running, enabled, label, tone };
}

function bucketOf(layout) {
  if (layout.running) return 'running';
  if (layout.enabled || layout.starting || layout.stopping) return 'pending';
  return 'stopped';
}

function isProblem(item) {
  if ((item.targets || []).some((target) => DANGER_STATES.has(target.state) || WARN_STATES.has(target.state))) {
    return true;
  }
  const rule = state.rules.find((entry) => entry.id === item.rule_id);
  return (rule?.triggers || []).some(triggerExpired);
}

function passesFilter(item, layout) {
  if (statusFilter === 'all') return true;
  if (statusFilter === 'problem') return isProblem(item);
  return bucketOf(layout) === statusFilter;
}

function resultTone(targets) {
  if (targets.some((target) => DANGER_STATES.has(target.state))) return 'is-failed';
  if (targets.some((target) => WARN_STATES.has(target.state))) return 'is-warning';
  if (targets.some((target) => PHASES[target.phase])) return 'is-active';
  return '';
}

function targetTone(target, phase) {
  if (phase) return 'is-active';
  if (DANGER_STATES.has(target.state)) return 'is-failed';
  if (WARN_STATES.has(target.state)) return 'is-warning';
  return '';
}

function clockNow() {
  return new Date().toLocaleTimeString('zh-CN', { hour12: false });
}

/** 关注度只给「小条 + 数字」的尺度感，文字解释交给 title，避免挤占结果列。 */
function attentionHtml(target) {
  const attention = target.attention || { weight: 0.5, observations: 0 };
  const rawWeight = Number(attention.weight);
  const weight = Number.isFinite(rawWeight) ? Math.min(1, Math.max(0, rawWeight)) : 0.5;
  const observations = Number(attention.observations) || 0;
  const observed = observations >= 1 ? String(Number(observations.toFixed(1))) : observations.toFixed(2);
  const note = observations > 0 ? `观测值 ${observed}` : '尚无观测';
  return `<span class="attention${observations > 0 ? '' : ' is-empty'}">
      <span class="attention-label">关注度</span>
      <span class="attention-bar" aria-hidden="true"><i style="width:${(weight * 100).toFixed(0)}%"></i></span>
      <b>${weight.toFixed(2)}</b>
      <span class="attention-note">${escapeHtml(note)}</span>
    </span>`;
}

function targetRow(item, target, running) {
  const key = `${item.binding_id}|${target.umo}`;
  const forcing = state.busyForcedTargets.has(key);
  const parts = targetParts(target.umo);
  const phase = PHASES[target.phase];
  const failed = DANGER_STATES.has(target.state);
  const resultText = phase || target.last_result || '-';
  const tone = targetTone(target, phase);
  const resultTitle = failed
    ? '决策模型判定失败，本次未发送消息；页面轮询期间会保留这条原因'
    : phase
      ? '本次执行正在推进，结果尚未产生'
      : '最近一次触发的处理结果';
  const forceTitle = running
    ? '立即对该会话执行一次完整判断（拦截 → 意向 → 投递）'
    : '绑定已停止，启动后才能手动触发';
  return `<li>
    <span class="target-id" title="${escapeHtml(target.umo)}">
      <b>${escapeHtml(parts.kind)}</b>
      <code>${escapeHtml(parts.session)}</code>
    </span>
    <span class="target-result ${tone}" title="${escapeHtml(resultTitle)}">${escapeHtml(resultText)}</span>
    <span class="target-meta">${attentionHtml(target)}</span>
    <em class="target-time" title="本插件在该会话最近一次成功自主发言的时间，普通对话回复不计入">${escapeHtml(formatStamp(target.last_fire_at))}</em>
    <span class="target-actions">
      <button class="btn small ghost" type="button" title="查看该会话的上下文缓冲（只读）"
        data-buffer-binding="${escapeHtml(item.binding_id)}"
        data-buffer-umo="${escapeHtml(target.umo)}">缓冲区</button>
      <button class="btn small ghost${forcing ? ' is-loading' : ''}" type="button"
        title="${escapeHtml(forceTitle)}" ${running && !forcing ? '' : 'disabled'}
        data-force-binding="${escapeHtml(item.binding_id)}"
        data-force-umo="${escapeHtml(target.umo)}">手动触发一次</button>
    </span>
  </li>`;
}

function statusCard(item) {
  const layout = lifecycleOf(item);
  const { request, stopping, starting, running, enabled } = layout;
  const busy = stopping || Boolean(request);
  const toggleLabel = stopping ? '停止收尾中…' : starting ? '取消启动' : enabled ? '停止' : '启动';
  const platform = platformParts(item.platform_id);
  const rule = state.rules.find((entry) => entry.id === item.rule_id);
  const expired = (rule?.triggers || []).filter(triggerExpired).length;
  const expiredNote = expired
    ? `<p class="hint error">${expired} 个触发入口已过期，不会再触发；到规则里改成将来的时间即可。</p>`
    : '';
  const targets = item.targets || [];
  const tone = resultTone(targets);
  const foldable = !running && targets.length > 1;
  // 运行中的绑定总是展开；停止态默认折叠，但只有一条会话时没有折叠按钮，必须直接显示
  const expanded = running || !foldable || expandedTargets.has(String(item.binding_id));
  const rows = targets.map((target) => targetRow(item, target, running)).join('');
  const toolbar = foldable
    ? `<div class="target-toolbar">
        <button class="btn small ghost" type="button" data-toggle-targets="${escapeHtml(item.binding_id)}">${expanded ? `收起 ${targets.length} 个会话` : `展开 ${targets.length} 个会话`}</button>
      </div>`
    : '';
  return `<article class="status-card ${running ? 'is-running' : 'is-stopped'}">
    <header>
      <div class="route">
        <b class="binding-name">${escapeHtml(item.binding_name || '未命名绑定')}</b>
        <p class="route-sub">
          <span class="bot-id">机器人「${escapeHtml(platform.id)}」</span>
          <span class="route-sep">→</span>
          <span>规则「${escapeHtml(item.rule_name || item.rule_id || '-')}」</span>
          <span class="route-dot">·</span>
          <span>${escapeHtml(selectionText(item.selection))}</span>
          <span class="route-dot">·</span>
          <span>${item.target_count} 个目标会话</span>
        </p>
      </div>
      <div class="status-actions">
        <span class="pill status-pill ${layout.tone}">${escapeHtml(layout.label)}</span>
        <button class="btn small ghost" type="button"
          title="把该绑定所有目标会话的关注度清回 0.50，历史判定不再影响挑选"
          data-reset-attention="${escapeHtml(item.binding_id)}">重置关注度</button>
        <button class="btn small ${running ? 'ghost' : 'primary'}${busy ? ' is-loading' : ''}"
          data-toggle-binding="${escapeHtml(item.binding_id)}"
          data-next-enabled="${starting || enabled ? 'false' : 'true'}"
          type="button"${stopping || (busy && !starting) ? ' disabled' : ''}>${toggleLabel}</button>
      </div>
    </header>
    <div class="status-grid">
      <div>
        <span>下次触发</span>
        <b>${escapeHtml(running ? formatMoment(item.next_fire_at) : '—')}</b>
        <span class="countdown" data-countdown-for="${escapeHtml(item.binding_id)}">${escapeHtml(running ? formatCountdown(item.next_fire_in) : '未调度')}</span>
      </div>
      <div>
        <span>上次触发</span>
        <b>${escapeHtml(formatMoment(item.last_fire_at))}</b>
      </div>
      <div>
        <span>上次结果</span>
        <b class="${tone}">${escapeHtml(item.last_result || '-')}</b>
      </div>
    </div>
    ${expiredNote}
    ${toolbar}
    ${expanded ? `<ul class="target-rows">${rows}</ul>` : ''}
  </article>`;
}

function renderSummary(items) {
  const summary = $('#status-summary');
  const counts = { running: 0, pending: 0, stopped: 0, targets: 0, problem: 0 };
  for (const item of items) {
    const layout = lifecycleOf(item);
    counts[bucketOf(layout)] += 1;
    counts.targets += Number(item.target_count) || 0;
    if (isProblem(item)) counts.problem += 1;
  }
  const chips = FILTERS.map(({ key, label }) => {
    const suffix = key === 'problem' && counts.problem ? ` ${counts.problem}` : '';
    return `<button class="filter-chip${statusFilter === key ? ' active' : ''}" type="button"
      aria-pressed="${statusFilter === key}" data-status-filter="${key}">${label}${suffix}</button>`;
  }).join('');
  summary.classList.remove('hidden');
  summary.innerHTML = `<div class="status-counts">
      <span class="summary-item"><b class="ok">${counts.running}</b>运行中</span>
      <span class="summary-item"><b class="pending">${counts.pending}</b>过渡中</span>
      <span class="summary-item"><b>${counts.stopped}</b>已停止</span>
      <span class="summary-item">${counts.targets} 个目标会话</span>
    </div>
    <div class="status-filters" role="group" aria-label="按运行状态筛选">${chips}</div>
    <span class="status-refreshed" data-status-refreshed title="最近一次拉取运行状态的时间">上次刷新 ${clockNow()}</span>`;
}

function statusFingerprint() {
  return JSON.stringify({
    enabled: state.meta?.enabled,
    filter: statusFilter,
    expanded: [...expandedTargets].sort(),
    busy: [...state.busyBindings.entries()].sort(),
    forced: [...state.busyForcedTargets].sort(),
    rules: state.rules.map((rule) => [rule.id, rule.triggers]),
    platforms: state.meta?.platforms || [],
    status: state.status.map(({ next_fire_in, ...rest }) => rest),
  });
}

/** 指纹不变时只刷新会随时间变化的倒计时与「上次刷新」，不整块重绘。 */
function patchLiveStatus() {
  const stamp = document.querySelector('[data-status-refreshed]');
  if (stamp) stamp.textContent = `上次刷新 ${clockNow()}`;
  for (const node of document.querySelectorAll('[data-countdown-for]')) {
    const item = state.status.find((entry) => String(entry.binding_id) === node.dataset.countdownFor);
    const running = item ? lifecycleOf(item).running : false;
    node.textContent = running ? formatCountdown(item.next_fire_in) : '未调度';
  }
}

export function renderStatus() {
  const container = $('#status-list');
  const summary = $('#status-summary');
  const fingerprint = statusFingerprint();
  if (fingerprint === lastFingerprint) {
    patchLiveStatus();
    return;
  }
  lastFingerprint = fingerprint;

  if (state.meta?.enabled === false) {
    summary.innerHTML = '';
    summary.classList.add('hidden');
    container.innerHTML =
      '<div class="empty">「启用自主发言」已关闭，调度未运行。规则与绑定仍可编辑。</div>';
    return;
  }
  if (!state.status.length) {
    summary.innerHTML = '';
    summary.classList.add('hidden');
    container.innerHTML = `<div class="empty">
      <p>还没有氛围组绑定。先到「氛围组绑定」为某个机器人应用新建一条。</p>
      <button class="btn small primary empty-action" type="button" data-section="bindings">去新建绑定</button>
    </div>`;
    return;
  }

  renderSummary(state.status);
  const visible = state.status.filter((item) => passesFilter(item, lifecycleOf(item)));
  if (!visible.length) {
    container.innerHTML = `<div class="empty">
      <p>没有符合当前筛选的绑定。</p>
      <button class="btn small ghost empty-action" type="button" data-status-filter="all">显示全部</button>
    </div>`;
    return;
  }

  const grouped = new Set(state.status.map((item) => item.platform_id)).size > 1;
  let html = '';
  let currentPlatform = null;
  for (const item of visible) {
    if (grouped && item.platform_id !== currentPlatform) {
      currentPlatform = item.platform_id;
      const platform = platformParts(currentPlatform);
      const count = visible.filter((entry) => entry.platform_id === currentPlatform).length;
      html += `<h3 class="status-group">
        <b class="bot-id">${escapeHtml(platform.id)}</b>
        ${platform.desc ? `<span class="bot-desc">${escapeHtml(platform.desc)}</span>` : ''}
        <em>${count} 条绑定</em>
      </h3>`;
    }
    html += statusCard(item);
  }
  container.innerHTML = html;
}

async function toggleBinding(bindingId, enabled) {
  const id = String(bindingId || '');
  if (!id) return;
  const pending = state.busyBindings.get(id);
  if (pending && !(pending.enabled && !enabled)) return;
  const version = (state.bindingToggleVersions.get(id) || 0) + 1;
  state.bindingToggleVersions.set(id, version);
  state.overviewRequest += 1;
  state.statusRequest += 1;
  state.busyBindings.set(id, { enabled, version });
  const current = () => state.bindingToggleVersions.get(id) === version;
  if (state.section === 'status') renderStatus();
  try {
    const data = await apiPost('bindings/toggle', { binding_id: id, enabled });
    if (!current()) return;
    await finishSaved(data, () => {
      if (!current()) return '';
      const binding = state.bindings.find((item) => String(item.id) === id);
      if (!binding || Boolean(binding.enabled) !== enabled) return '绑定状态已更新';
      return enabled ? '绑定已启动' : '绑定已停止（未完成的启动已取消）';
    }, current);
  } catch (error) {
    if (current()) toast(error.message || '操作失败', true);
  } finally {
    if (current()) state.busyBindings.delete(id);
    if (state.section === 'status') renderStatus();
  }
}

/**
 * 手动触发一个目标会话：先弹窗说明，确认后才真正执行。
 * 与定时触发共用同一条链路（拦截 → 意向 → 投递），结果回显在 toast。
 */
async function forceTrigger(bindingId, umo) {
  const id = String(bindingId || '');
  const target = String(umo || '');
  if (!id || !target) return;
  const key = `${id}|${target}`;
  if (state.busyForcedTargets.has(key)) return;

  const item = state.status.find((entry) => String(entry.binding_id) === id);
  if (!item || !(item.ready ?? item.running)) {
    toast('绑定已停止，启动后才能手动触发', true);
    return;
  }

  const confirmed = await askConfirm(
    [
      `会话：${target}`,
      `绑定：${item.binding_name || '未命名绑定'}`,
      `规则：${item.rule_name || '-'}`,
      '',
      '将立即执行一次：拦截检查 → 发言意向 → 投递。',
      '触发入口与挑选策略会被跳过；若被拦截或判定为不发言，本次不会发送任何消息。',
    ].join('\n'),
    '手动触发一次',
    '确定触发',
    { primary: true },
  );
  if (!confirmed) return;

  state.busyForcedTargets.add(key);
  if (state.section === 'status') renderStatus();
  try {
    const result = await apiPost('bindings/trigger', { binding_id: id, umo: target });
    toast(
      result?.spoke || ['unknown', 'cancelled', 'busy'].includes(result?.state)
        ? result.reason || '已发言'
        : `未发言：${result?.reason || '无结果'}`,
    );
  } catch (error) {
    toast(error.message || '手动触发失败', true);
  } finally {
    state.busyForcedTargets.delete(key);
    if (state.section === 'status') {
      await loadStatus().catch(() => {});
      renderStatus();
    }
  }
}

/**
 * 重置一条绑定的关注度：属于运行期画像而非配置，因此绑定停止时也能点。
 * 关注度会回到 0.50（尚无观测），此后按新的判定重新累积。
 */
async function resetAttention(bindingId) {
  const id = String(bindingId || '');
  if (!id) return;
  const confirmed = await askConfirm(
    [
      '该绑定所有目标会话的关注度都会回到 0.50（尚无观测）。',
      '历史判定不再影响挑选，之后按新的判定重新累积；其余运行状态不受影响。',
    ].join('\n'),
    '重置关注度',
    '确定重置',
    { primary: true },
  );
  if (!confirmed) return;
  try {
    await apiPost('bindings/attention/reset', { binding_id: id });
    toast('关注度已重置');
  } catch (error) {
    toast(error.message || '重置关注度失败', true);
  } finally {
    if (state.section === 'status') await loadStatus().catch(() => {});
  }
}

on('section', (name) => {
  if (name !== 'status') {
    stopStatusPolling();
    return;
  }
  loadStatus().catch((error) => toast(error.message || '读取运行状态失败', true));
  startStatusPolling();
});

// 保存类操作改动了概览数据，停在运行管理页时要补一次状态。
on('afterSave', async () => {
  if (state.section === 'status') await loadStatus().catch(() => {});
});

export const actions = {
  'data:toggleBinding': (bindingId, button) =>
    toggleBinding(bindingId, button.dataset.nextEnabled === 'true'),
  'data:forceBinding': (bindingId, button) => forceTrigger(bindingId, button.dataset.forceUmo),
  'data:resetAttention': (bindingId) => resetAttention(bindingId),
  // 停止态卡片的会话明细默认折叠，展开状态记在内存里，轮询重绘后不回弹。
  'data:toggleTargets': (bindingId) => {
    const id = String(bindingId || '');
    if (!id) return;
    if (expandedTargets.has(id)) expandedTargets.delete(id);
    else expandedTargets.add(id);
    if (state.section === 'status') renderStatus();
  },
  'data:statusFilter': (value) => {
    if (!FILTERS.some((entry) => entry.key === value)) return;
    statusFilter = value;
    if (state.section === 'status') renderStatus();
  },
};
