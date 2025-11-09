// 简易 i18n 配置与语言检测
type Locale = "zh" | "en";

const translations: Record<Locale, Record<string, string>> = {
  zh: {
    welcome:
      "您好！我是 DeepAnalyze-8B，您的自主数据科学助手。请上传数据文件，让我们一起探索并分析它！",
    assistant: "助手",
    busyClear: "执行中，暂时无法清空",
    cleared: "已清空聊天",
    busyExport: "执行中，暂时无法导出",
    exportFailed: "导出失败",
  },
  en: {
    welcome:
      "Hello! I'm DeepAnalyze-8B, your autonomous data science assistant. Upload your data and let's explore it together!",
    assistant: "Assistant",
    busyClear: "Currently running, cannot clear yet",
    cleared: "Chat cleared",
    busyExport: "Currently running, cannot export yet",
    exportFailed: "Export failed",
  },
};

let currentLocale: Locale = "zh";

export function detectLocale(): Locale {
  try {
    if (typeof window !== "undefined") {
      const saved = window.localStorage.getItem("locale");
      if (saved === "zh" || saved === "en") return saved as Locale;
      const nav = (navigator.language || (navigator as any).userLanguage || "").toLowerCase();
      if (nav.startsWith("zh")) return "zh";
    }
  } catch {}
  // SSR 默认中文，避免中英不一致导致 hydration 警告
  return "zh";
}

export function setLocale(locale: Locale) {
  currentLocale = locale;
  try {
    if (typeof window !== "undefined") {
      window.localStorage.setItem("locale", locale);
    }
  } catch {}
}

export function getLocale(): Locale {
  if (!currentLocale) currentLocale = detectLocale();
  return currentLocale;
}

export function t(key: string): string {
  const loc = getLocale();
  const pack = translations[loc] || translations.zh;
  return pack[key] ?? key;
}

// 初始化 locale
currentLocale = detectLocale();