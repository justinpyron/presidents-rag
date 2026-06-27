// Enter submits, Shift+Enter inserts a newline. Uses event delegation so it
// keeps working as the composer is re-rendered.
document.addEventListener("keydown", function (e) {
  const el = e.target;
  if (el && el.classList && el.classList.contains("composer-input")) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const send = document.getElementById("send-btn");
      if (send) send.click();
    }
  }
});
