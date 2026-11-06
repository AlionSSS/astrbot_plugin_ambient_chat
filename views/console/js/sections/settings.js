/** 更多设置页：运行参数、自定义决策模型与运行信息。 */
import { apiPost } from '../api.js';
import { finishSaved } from '../shell.js';
import { on, state } from '../store.js';
import { askConfirm, askInfo, toast } from '../ui.js';
import { $, clone, escapeHtml } from '../util.js';

export function renderSettings() {
  const settings = state.meta?.settings;
  if (!settings) return;
  $('#setting-concurrency').value = settings.concurrency;
  $('#setting-interval').value = settings.send_interval_seconds;
  $('#setting-buffer').value = settings.buffer_size;
  $('#setting-ignored-senders').value = (
    settings.buffer_ignored_sender_ids || []
  ).join(', ');
  $('#setting-cache-images').checked = Boolean(settings.cache_images);
  $('#setting-image-edge').value = settings.image_max_edge;
  $('#setting-image-quality').value = settings.image_quality;
  $('#setting-decision-debug').checked = Boolean(settings.decision_debug_enabled);
  state.decisionModels = clone(settings.decision_models || []);
  state.decisionTestResults = {};
  renderDecisionModels();
  $('#info-enabled').textContent = state.meta.enabled ? '已开启（调度运行中）' : '已关闭（仅暂停调度）';
  $('#info-timezone').textContent = state.meta.timezone || '-';
  $('#info-path').textContent = state.meta.config_path || '-';
  $('#info-media-path').textContent = state.meta.media_path || '-';
  $('#info-schema').textContent = `v${state.meta.schema_version ?? '-'}`;
  $('#info-error').textContent = state.meta.load_error || '无';
}

function blankDecisionModel() {
  return {
    id: '',
    name: '',
    base_url: '',
    model: '',
    api_key: '',
  };
}

/**
 * 自定义决策模型卡片。输入用 data-dm-index / data-dm-field 回写到 state，
 * 保存时整体提交（后端负责裁剪与校验）。
 */
function renderDecisionModels() {
  const container = $('#decision-models');
  if (!container) return;
  container.innerHTML = state.decisionModels
    .map(
      (model, index) => `<div class="card">
      <div class="card-head">
        <b>${escapeHtml(model.name || model.model || `决策模型 ${index + 1}`)}</b>
        <span class="card-actions">
          <button class="btn small ghost" data-test-decision-model="${index}" type="button"
            title="用一次请求同时测试 noul / choice / score">测试</button>
          <button class="btn small danger" data-remove-decision-model="${index}" type="button">删除</button>
        </span>
      </div>
      <div class="grid">
        <label class="field"><span>名称（可选）</span>
          <input type="text" maxlength="40" data-dm-index="${index}" data-dm-field="name"
            value="${escapeHtml(model.name)}" placeholder="例如：Jev决策" /></label>
        <label class="field"><span>模型 ID</span>
          <input type="text" maxlength="100" data-dm-index="${index}" data-dm-field="model"
            value="${escapeHtml(model.model)}" placeholder="jev-latest" /></label>
        <label class="field"><span>接口地址（Base URL）</span>
          <input type="text" maxlength="500" data-dm-index="${index}" data-dm-field="base_url"
            value="${escapeHtml(model.base_url)}" placeholder="https://api.example.com/v1" /></label>
        <div class="field"><span>API Key</span>
          <span class="secret-field">
            <input type="password" autocomplete="new-password" data-dm-index="${index}" data-dm-field="api_key"
              value="${escapeHtml(model.api_key)}" />
            <button class="secret-toggle" data-toggle-secret="1" type="button"
              aria-label="显示 API Key" title="显示 API Key">
              <svg class="icon-eye" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
              <svg class="icon-eye-off hidden" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
                <line x1="1" y1="1" x2="23" y2="23" />
              </svg>
            </button>
          </span>
        </div>
      </div>
      <div class="decision-test-result" data-dm-result="${index}"></div>
    </div>`,
    )
    .join('');
  const empty = $('#decision-models-empty');
  if (empty) empty.classList.toggle('hidden', state.decisionModels.length > 0);
  for (const key of Object.keys(state.decisionTestResults)) {
    renderDecisionTestResult(Number(key));
  }
}

function percentText(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : '-';
}

function formatDecisionTest(test) {
  if (test.status === 'loading') return '<p class="hint">测试中…</p>';
  if (test.status === 'error') {
    return `<p class="hint error">测试失败：${escapeHtml(test.message)}</p>`;
  }
  const result = test.result || {};
  const answers = result.answers || {};
  const lines = [];
  if (answers.noul) {
    lines.push(`noul：${percentText(answers.noul.probability)} 为真`);
  }
  if (answers.choice) {
    lines.push(
      `choice：${escapeHtml(answers.choice.choice)}（${percentText(answers.choice.confidence)}）`,
    );
  }
  if (answers.score) {
    lines.push(
      `score：${Number(answers.score.score).toFixed(2)}（${percentText(answers.score.confidence)}）`,
    );
  }
  const usage = result.usage
    ? `，tokens ${result.usage.input_tokens}/${result.usage.output_tokens}`
    : '';
  return `<p class="hint">✅ 三种题型均通过（${Number(result.elapsed_ms) || 0} ms${usage}）</p>
    <p class="hint">${lines.map((line) => escapeHtml(line)).join('　·　')}</p>
    <p class="hint">${escapeHtml(result.endpoint || '')}</p>`;
}

/** 只重绘该卡片的结果区，避免整块卡片重建导致输入框失焦。 */
function renderDecisionTestResult(index) {
  const node = document.querySelector(`[data-dm-result="${index}"]`);
  if (!node) return;
  const test = state.decisionTestResults[index];
  node.innerHTML = test ? formatDecisionTest(test) : '';
}

async function testDecisionModel(index) {
  const model = state.decisionModels[index];
  if (!model) return;
  if (!String(model.base_url || '').trim() || !String(model.model || '').trim()) {
    toast('请先填写接口地址与模型 ID', true);
    return;
  }
  state.decisionTestResults[index] = { status: 'loading' };
  renderDecisionTestResult(index);
  const button = document.querySelector(`[data-test-decision-model="${index}"]`);
  if (button) button.disabled = true;
  try {
    const data = await apiPost('decision_models/test', { model });
    state.decisionTestResults[index] = { status: 'ok', result: data.result };
  } catch (error) {
    state.decisionTestResults[index] = {
      status: 'error',
      message: error.message || '测试失败',
    };
  } finally {
    if (button) button.disabled = false;
    renderDecisionTestResult(index);
  }
}

/** 用户 ID 输入框：逗号（中英文）与空白都算分隔符，去重后提交。 */
function parseSenderIds(value) {
  const result = [];
  for (const item of String(value || '').split(/[,，\s]+/)) {
    const id = item.trim();
    if (id && !result.includes(id)) result.push(id);
  }
  return result;
}

async function saveSettings() {
  const payload = {
    settings: {
      concurrency: Number($('#setting-concurrency').value),
      send_interval_seconds: Number($('#setting-interval').value),
      buffer_size: Number($('#setting-buffer').value),
      buffer_ignored_sender_ids: parseSenderIds(
        $('#setting-ignored-senders').value,
      ),
      cache_images: $('#setting-cache-images').checked,
      image_max_edge: Number($('#setting-image-edge').value),
      image_quality: Number($('#setting-image-quality').value),
      decision_debug_enabled: $('#setting-decision-debug').checked,
      decision_models: clone(state.decisionModels),
      // 后端按整对象解析，漏传会重置成默认值（false），导致再次弹提示。
      cache_images_advice_seen: Boolean(
        state.meta?.settings?.cache_images_advice_seen,
      ),
    },
  };
  try {
    const data = await apiPost('settings/save', payload);
    await finishSaved(data, '设置已保存');
  } catch (error) {
    toast(error.message || '保存失败', true);
  }
}

const CACHE_IMAGES_ADVICE =
  '建议开启图片缓存：开启后，支持多模态的对话模型能看到最近对话里的图片。' +
  '代价是图片会转存到插件数据目录并持续占用磁盘空间，消息被挤出缓冲时才会删除；' +
  '落盘时会按「图片最长边」「图片压缩质量」重编码，画质会有损失；' +
  '图片还会随请求一起发送，模型每次调用的 token 消耗会明显增加。' +
  '关闭缓存或插件卸载时会清空已缓存的图片。';

const CACHE_IMAGES_ADVICE_TITLE = '图片缓存建议开启（默认已开启）';

/** 正在等待提示结果：快速切页时不重复弹窗。 */
let cacheImagesAdvicePending = false;

/**
 * 首次进入设置页时提示一次「建议开启」。
 *
 * 提示状态记在 `settings.json` 的 `settings.cache_images_advice_seen`，跨会话只提示一次；
 * 写入失败时保持未提示，下次进入设置页再弹。已开启只给「知道了」，未开启才提供「开启」。
 */
async function maybeShowCacheImagesAdvice() {
  if (cacheImagesAdvicePending) return;
  if (state.meta?.settings?.cache_images_advice_seen) return;
  const checkbox = $('#setting-cache-images');
  if (!checkbox) return;
  cacheImagesAdvicePending = true;
  try {
    if (checkbox.checked) {
      await askInfo(CACHE_IMAGES_ADVICE, CACHE_IMAGES_ADVICE_TITLE);
    } else {
      const confirmed = await askConfirm(
        CACHE_IMAGES_ADVICE,
        CACHE_IMAGES_ADVICE_TITLE,
        '开启',
        { primary: true },
      );
      if (confirmed) checkbox.checked = true;
    }
    await apiPost('settings/advice_seen');
    if (state.meta?.settings) {
      state.meta.settings.cache_images_advice_seen = true;
    }
  } catch {
    // 标记没写进去：保持未提示，下次进入设置页再弹。
  } finally {
    cacheImagesAdvicePending = false;
  }
}

/** 自定义决策模型的输入回写到 state（后端保存时做裁剪与校验）。 */
function syncDecisionModelField(input) {
  const index = Number(input.dataset.dmIndex);
  const field = input.dataset.dmField;
  if (!field || !Number.isInteger(index)) return;
  const model = state.decisionModels[index];
  if (!model) return;
  model[field] = input.type === 'checkbox' ? input.checked : input.value;
}

for (const eventName of ['input', 'change']) {
  document.addEventListener(eventName, (event) => {
    if (!event.target.closest('#decision-models')) return;
    syncDecisionModelField(event.target);
  });
}

/** 明文/密文切换：浏览器原生的「眼睛」在失焦后会消失，这里自己控制。 */
function toggleSecret(button) {
  const input = button.closest('.secret-field')?.querySelector('input');
  if (!input) return;
  const reveal = input.type === 'password';
  input.type = reveal ? 'text' : 'password';
  button.querySelector('.icon-eye')?.classList.toggle('hidden', reveal);
  button.querySelector('.icon-eye-off')?.classList.toggle('hidden', !reveal);
  button.title = reveal ? '隐藏 API Key' : '显示 API Key';
  button.setAttribute('aria-label', button.title);
}

on('overview', () => renderSettings());
on('section', (name) => {
  if (name !== 'settings') return;
  renderSettings();
  maybeShowCacheImagesAdvice();
});

export const actions = {
  'btn-save-settings': () => saveSettings(),
  'btn-add-decision-model': () => {
    state.decisionModels.push(blankDecisionModel());
    renderDecisionModels();
  },
  'data:toggleSecret': (_value, button) => toggleSecret(button),
  'data:testDecisionModel': (index) => testDecisionModel(Number(index)),
  'data:removeDecisionModel': (index) => {
    state.decisionModels.splice(Number(index), 1);
    // 序号会前移，旧结果不再对得上，直接清空。
    state.decisionTestResults = {};
    renderDecisionModels();
  },
};
