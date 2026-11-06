/** 全局的轻量 UI：toast、模态对话框、标题旁的「详细说明」卡片。 */
import { $ } from './util.js';

const TOAST_DURATION = 3200;

/** 重新开始倒计时；鼠标悬停在提示上时由 mouseenter 清掉。 */
function startToastCountdown() {
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => $('#toast').classList.add('hidden'), TOAST_DURATION);
}

export function toast(message, bad = false) {
  const node = $('#toast');
  node.textContent = message;
  node.classList.remove('hidden');
  node.style.background = bad ? 'var(--danger)' : 'var(--text)';
  clearTimeout(toast._timer);
  // 悬停中不启动倒计时，等 mouseleave 再从头计时。
  if (!node.matches(':hover')) startToastCountdown();
}

// 鼠标停在提示上就不要消失，方便用户慢慢看完。
$('#toast').addEventListener('mouseenter', () => clearTimeout(toast._timer));
$('#toast').addEventListener('mouseleave', (event) => {
  if (!event.currentTarget.classList.contains('hidden')) startToastCountdown();
});

/**
 * 页面内模态框。
 * 插件视图运行在 sandbox 没有 allow-modals 的 iframe 里，window.confirm() 会被
 * 浏览器直接忽略（永远返回 false），所以不能依赖它。
 *
 * `cancellable` 区分两类用途：确认型（有「取消」）与只读信息型（只有「知道了」）。
 * 两种形态复用同一套 DOM，关闭时恢复成「删除确认」的默认外观。
 */
export function askDialog({ message, title = '确认操作', okLabel = '确认删除', cancellable = true, primary = false }) {
  return new Promise((resolve) => {
    const dialog = $('#dialog');
    const okButton = $('#dialog-ok');
    const cancelButton = $('#dialog-cancel');

    $('#dialog-title').textContent = title;
    $('#dialog-message').textContent = message;
    okButton.textContent = okLabel;
    // 非破坏性动作用主色，避免「手动触发」看起来像删除。
    okButton.classList.toggle('primary', primary || !cancellable);
    okButton.classList.toggle('danger', !primary && cancellable);
    cancelButton.classList.toggle('hidden', !cancellable);
    dialog.classList.remove('hidden');

    const finish = (result) => {
      dialog.classList.add('hidden');
      cancelButton.classList.remove('hidden');
      okButton.classList.remove('primary');
      okButton.classList.add('danger');
      okButton.removeEventListener('click', onOk);
      cancelButton.removeEventListener('click', onCancel);
      document.removeEventListener('keydown', onKey);
      resolve(result);
    };
    const onOk = () => finish(true);
    const onCancel = () => finish(false);
    const onKey = (event) => {
      if (event.key === 'Escape') finish(false);
    };

    okButton.addEventListener('click', onOk);
    if (cancellable) cancelButton.addEventListener('click', onCancel);
    document.addEventListener('keydown', onKey);
    okButton.focus();
  });
}

/** 确认框：默认「删除」语义（危险色 + 有取消）。 */
export function askConfirm(message, title = '确认操作', okLabel = '确认删除', options = {}) {
  return askDialog({
    message,
    title,
    okLabel,
    cancellable: true,
    primary: Boolean(options.primary),
  });
}

/** 只读信息框（无「取消」，单个「知道了」）。 */
export function askInfo(message, title = '提示', okLabel = '知道了') {
  return askDialog({ message, title, okLabel, cancellable: false });
}

/* ------------------------------------------- 标题旁「详细说明」信息图标
   hover / 键盘聚焦由 CSS 负责，这里只管「点击钉住」，方便触屏与慢慢看。 */
export function setHelpOpen(help, open) {
  help.classList.toggle('is-open', open);
  const trigger = help.querySelector('.help-icon');
  if (trigger) trigger.setAttribute('aria-expanded', String(open));
}

/** 收起所有被点击钉住的说明卡片，返回是否收过。 */
export function closePinnedHelp() {
  const pinned = document.querySelectorAll('.section-help.is-open');
  pinned.forEach((help) => setHelpOpen(help, false));
  return pinned.length > 0;
}
