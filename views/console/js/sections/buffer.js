/**
 * 上下文缓冲弹窗：一次点击 = 一份只读快照。
 * 缓冲只在内存里，运行状态轮询（10 秒）不会顺带刷新它——想看最新的点「刷新」，
 * 因此这里不注册任何定时器。
 */
import { apiGet } from '../api.js';
import { state } from '../store.js';
import { $, escapeHtml, kindLabel } from '../util.js';

export function openBufferModal(bindingId, umo) {
  const id = String(bindingId || '');
  const only = String(umo || '');
  if (!id || !only) return;
  state.bufferBindingId = id;
  state.bufferUmo = only;
  state.bufferView = { loading: true, error: '', data: null };
  $('#buffer-modal').classList.remove('hidden');
  renderBufferModal();
  loadBuffer().catch(() => {});
}

export function closeBufferModal() {
  $('#buffer-modal').classList.add('hidden');
  state.bufferBindingId = null;
  state.bufferUmo = '';
  state.bufferView = null;
}

async function loadBuffer() {
  const bindingId = state.bufferBindingId;
  const umo = state.bufferUmo;
  if (!bindingId || !umo) return;
  try {
    const data = await apiGet('buffer', { binding_id: bindingId, umo });
    // 期间弹窗可能已被关闭或换到别的会话，旧响应直接丢弃。
    if (state.bufferBindingId !== bindingId || state.bufferUmo !== umo) return;
    state.bufferView = { loading: false, error: '', data };
  } catch (error) {
    if (state.bufferBindingId !== bindingId || state.bufferUmo !== umo) return;
    state.bufferView = {
      loading: false,
      error: error.message || '读取缓冲区失败',
      data: null,
    };
  }
  renderBufferModal();
}

function formatBytes(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MiB`;
}

/** 条数与媒体占用：截断时说明只显示最近 N 条。 */
function bufferStatsText(session, previewLimit) {
  const count = session.truncated
    ? `共 ${session.count} 条 · 仅显示最近 ${Math.min(previewLimit, session.count)} 条`
    : `共 ${session.count} 条消息`;
  const media = session.media_count
    ? ` · 图片 ${session.media_count} 张 · ${formatBytes(session.media_bytes)}`
    : '';
  const audio = session.audio_count
    ? ` · 语音 ${session.audio_count} 段 · ${formatBytes(session.audio_bytes)}`
    : '';
  return `${count}${media}${audio}`;
}

/** 消息列表：时间 / 发送者名字 / 发送者 ID / 正文四列对齐，按时间正序，最新在底部。 */
function bufferMessagesHtml(session) {
  const rows = session.messages || [];
  if (!rows.length) {
    return '<div class="buffer-empty">这个会话的缓冲里还没有消息。停止绑定会清空缓冲，启动后会重新预热。</div>';
  }
  return `<ol class="buffer-messages">${rows
    .map(
      (message) => `<li class="${message.self ? 'self' : ''}">
        <span class="buffer-time">${escapeHtml(message.at)}</span>
        <span class="buffer-sender">${escapeHtml(message.sender_name || '某人')}</span>
        <span class="buffer-sender-id">${escapeHtml(message.sender_id || '-')}</span>
        <span class="buffer-text" title="${escapeHtml(message.text || '（空消息）')}">${escapeHtml(message.text || '（空消息）')}</span>
      </li>`,
    )
    .join('')}</ol>`;
}

function renderBufferModal() {
  const view = state.bufferView;
  if (!view) return;
  const data = view.data || {};
  const title = $('#buffer-modal-title');
  const body = $('#buffer-body');

  // 弹窗只对应一个目标会话：标题固定给出 UMO，能拿到会话名时再补在后面。
  const session = (data.sessions || [])[0] || null;
  title.textContent = session
    ? `上下文缓冲 · ${session.umo}${session.session_name ? ` · ${session.session_name}` : ''}`
    : '上下文缓冲';

  if (view.loading) {
    body.innerHTML = '<p class="hint">读取中…</p>';
    return;
  }
  if (view.error) {
    body.innerHTML = `<p class="hint error">${escapeHtml(view.error)}</p>`;
    return;
  }
  if (!session) {
    body.innerHTML = '<div class="buffer-empty">这条绑定还没有目标会话。</div>';
    return;
  }
  // 正文首行：绑定范围在左；上限与条数 / 图片占用是同类信息，一起靠右。
  body.innerHTML =
    `<p class="buffer-scope">
      <span class="pill">${kindLabel(session.message_type)}</span>
      <span>绑定「${escapeHtml(data.binding_name || data.binding_id)}」</span>
      <span class="buffer-scope-stats">每会话上限 ${data.buffer_maxlen} 条 · ${bufferStatsText(session, data.preview_limit)}</span>
    </p>` + bufferMessagesHtml(session);
  // 打开 / 刷新后直接落在最新一条上：排查「刚才怎么没接话」不必先滚到底。
  // 滚动区是消息列表本身（顶部绑定信息固定不滚），所以滚列表而不是弹窗正文。
  const list = body.querySelector('.buffer-messages');
  if (list) list.scrollTop = list.scrollHeight;
}

export const actions = {
  'data:bufferBinding': (bindingId, button) => openBufferModal(bindingId, button.dataset.bufferUmo),
  'btn-buffer-refresh': () => loadBuffer(),
  'btn-close-buffer': () => closeBufferModal(),
};
