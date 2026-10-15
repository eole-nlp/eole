"use strict";
window.EoleUI = {
  chooseModel(state, view, candidates, current) {
    const ids = candidates.map((m) => m.id);
    if (ids.includes(current)) return current;
    let saved;
    try {
      saved = localStorage.getItem(`eole.workspace.model.${view}`);
    } catch {}
    return (
      [
        saved,
        state.workspace?.models?.[view],
        view === "translate"
          ? candidates.find((m) => m.translation?.preferred)?.id
          : null,
        ids[0],
      ].find((id) => ids.includes(id)) || ""
    );
  },
  rememberModel(view, id) {
    try {
      localStorage.setItem(`eole.workspace.model.${view}`, id);
    } catch {}
  },
  escape(value) {
    return String(value ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  },
  async *events(response) {
    if (!response.ok) {
      const body = await response.json();
      throw new Error(
        typeof body.detail === "string"
          ? body.detail
          : "The request settings were rejected.",
      );
    }
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let position;
        while ((position = buffer.indexOf("\n\n")) >= 0) {
          const frame = buffer.slice(0, position);
          buffer = buffer.slice(position + 2);
          const data = frame
            .split("\n")
            .filter((l) => l.startsWith("data:"))
            .map((l) => l.slice(5).trimStart())
            .join("\n");
          if (data) yield JSON.parse(data);
        }
        if (done) break;
      }
    } finally {
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  },
  download(name, content, type = "text/plain") {
    const url = URL.createObjectURL(new Blob([content], { type }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  },
};
