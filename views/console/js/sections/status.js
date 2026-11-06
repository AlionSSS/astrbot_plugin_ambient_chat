/** 运行管理页：轮询运行状态，提供启停与手动触发。 */
import { apiGet, apiPost } from '../api.js';
import { platformParts, selectionText, triggerExpired } from '../model.js';
import { finishSaved, renderBanner, renderNav } from '../shell.js';
import { on, state } from '../store.js';
import { askConfirm, toast } from '../ui.js';
import { $, escapeHtml, formatCountdown } from '../util.js';

let statusTimer = null;

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

export function renderStatus() {
  const container = $('#status-list');
  if (state.meta?.enabled === false) {
    container.innerHTML =
      '<div class="empty">「启用自主发言」已关闭，调度未运行。规则与绑定仍可编辑。</div>';
    return;
  }
  if (!state.status.length) {
    container.innerHTML =
      '<div class="empty">还没有 氛围组绑定。到「氛围组绑定」里新建一条即可。</div>';
    return;
  }
  container.innerHTML = state.status
    .map((item) => {
      const request = state.busyBindings.get(String(item.binding_id));
      const stopping = item.lifecycle === 'stopping' || request?.enabled === false;
      const starting = !stopping && (item.lifecycle === 'starting' || request?.enabled === true);
      const running = !stopping && !starting && (item.ready ?? Boolean(item.running));
      const enabled = Boolean(item.enabled);
      const busy = stopping || Boolean(request);
      const toggleLabel = stopping ? '停止收尾中…' : starting ? '取消启动' : enabled ? '停止' : '启动';
      const lifecycleLabel = stopping ? '停止收尾中' : starting ? '启动预热中'
        : running ? '运行中' : enabled ? '等待应用' : '已停止';
      const platform = platformParts(item.platform_id);
      const rule = state.rules.find((entry) => entry.id === item.rule_id);
      const expired = (rule?.triggers || []).filter(triggerExpired).length;
      const expiredNote = expired
        ? `<p class="hint error">${expired} 个入口触发已过期，不会再触发；到规则里改成将来的时间即可。</p>`
        : '';
      const targets = (item.targets || [])
        .map((target) => {
          const forcing = state.busyForcedTargets.has(`${item.binding_id}|${target.umo}`);
          const forceTitle = running
            ? '立即对该会话执行一次完整判断（拦截 → 意向 → 投递）'
            : '绑定已停止，启动后才能手动触发';
          // 决策失败（含平台内容审核拒绝）不是普通跳过：单独标红，页面直接给出原因。
          const failed = target.state === 'decision_failed';
          const phase = ({ queued: '等待并发名额', judging: '判断中', generating: '生成中', pacing: '等待发送时机', sending: '发送中', recording: '记录中' })[target.phase];
          const resultText = phase || target.last_result || '-';
          return `<li>
            <code>${escapeHtml(target.umo)}</code>
            <span class="${failed && !phase ? 'failed' : 'muted'}"${failed ? ' title="决策模型判定失败，本次未发送消息；页面轮询期间会保留这条原因"' : ''}>${escapeHtml(resultText)}</span>
            <em title="本插件在该会话最近一次成功自主发言的时间，普通对话回复不计入">${escapeHtml(target.last_fire_at || '-')}</em>
            <button class="btn small ghost" type="button" title="查看该会话的上下文缓冲（只读）"
              data-buffer-binding="${escapeHtml(item.binding_id)}"
              data-buffer-umo="${escapeHtml(target.umo)}">缓冲区</button>
            <button class="btn small ghost${forcing ? ' is-loading' : ''}" type="button"
              title="${escapeHtml(forceTitle)}" ${running && !forcing ? '' : 'disabled'}
              data-force-binding="${escapeHtml(item.binding_id)}"
              data-force-umo="${escapeHtml(target.umo)}">手动触发一次</button>
          </li>`;
        })
        .join('');
      return `<article class="status-card ${running ? '' : 'stopped'}">
        <header>
          <div class="route">
            <b class="bot-id">${escapeHtml(platform.id)}</b>
            <span class="route-sep">→</span>
            <span>${escapeHtml(item.binding_name || '未命名绑定')}</span>
            <span class="rule-pill">规则「${escapeHtml(item.rule_name)}」</span>
            <span class="pill ${running ? 'on' : 'off'}">${escapeHtml(lifecycleLabel)}</span>
          </div>
          <div class="status-actions">
            <span class="muted">${escapeHtml(selectionText(item.selection))} · ${item.target_count} 个目标会话</span>
            <button class="btn small ${running ? 'ghost' : 'primary'}${busy ? ' is-loading' : ''}"
              data-toggle-binding="${escapeHtml(item.binding_id)}"
              data-next-enabled="${starting || enabled ? 'false' : 'true'}"
              type="button"${stopping || (busy && !starting) ? ' disabled' : ''}>${toggleLabel}</button>
          </div>
        </header>
        <div class="status-grid">
          <div><span>下次触发</span><b>${escapeHtml(running ? item.next_fire_at || '-' : '—')}</b>
            <span>${escapeHtml(running ? formatCountdown(item.next_fire_in) : '未调度')}</span></div>
          <div><span>上次触发</span><b>${escapeHtml(item.last_fire_at || '-')}</b></div>
          <div><span>上次结果</span><b>${escapeHtml(item.last_result || '-')}</b></div>
        </div>
        ${expiredNote}
        <ul class="target-rows">${targets}</ul>
      </article>`;
    })
    .join('');
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
      '入口触发与挑选策略会被跳过；若被拦截或判定为不发言，本次不会发送任何消息。',
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
};
