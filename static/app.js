const input = document.querySelector("#photo-input");
const dropzone = document.querySelector("#dropzone");
const uploadView = document.querySelector("#upload-view");
const previewView = document.querySelector("#preview-view");
const loadingView = document.querySelector("#loading-view");
const resultView = document.querySelector("#result-view");
const preview = document.querySelector("#preview");
const checkButton = document.querySelector("#check-button");
const changeButton = document.querySelector("#change-button");
const retryButton = document.querySelector("#retry-button");
const continueButton = document.querySelector("#continue-button");
let selectedFile = null;
let objectUrl = null;

function show(view) {
  [uploadView, previewView, loadingView, resultView].forEach((item) => {
    item.hidden = item !== view;
  });
}

function selectFile(file) {
  if (!file) return;
  const validTypes = ["image/jpeg", "image/png", "image/webp"];
  if (!validTypes.includes(file.type)) {
    window.alert("Please choose a JPEG, PNG, or WebP image.");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    window.alert("Please choose an image smaller than 8 MB.");
    return;
  }
  selectedFile = file;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  preview.src = objectUrl;
  show(previewView);
}

input.addEventListener("change", () => selectFile(input.files[0]));
["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
});
dropzone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

function reset() {
  selectedFile = null;
  input.value = "";
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  show(uploadView);
}

changeButton.addEventListener("click", reset);
retryButton.addEventListener("click", reset);

function renderResult(result) {
  const banner = document.querySelector("#result-banner");
  const icon = document.querySelector("#result-icon");
  const kicker = document.querySelector("#result-kicker");
  const title = document.querySelector("#result-title");
  const summary = document.querySelector("#result-summary");
  const list = document.querySelector("#check-list");
  const notes = document.querySelector("#result-notes");

  banner.className = "result-banner";
  if (result.status === "valid") {
    icon.textContent = "✓";
    kicker.textContent = "PHOTO ACCEPTED";
    title.textContent = "Your photo is ready";
    summary.textContent = "It meets all automatic registration photo checks.";
    continueButton.hidden = false;
  } else if (result.status === "invalid") {
    banner.classList.add("invalid");
    icon.textContent = "×";
    kicker.textContent = "PHOTO NOT ACCEPTED";
    title.textContent = "Please use another photo";
    summary.textContent = "Fix the items marked below, then check a new photo.";
    continueButton.hidden = true;
  } else {
    banner.classList.add("review");
    icon.textContent = "!";
    kicker.textContent = "REVIEW NEEDED";
    title.textContent = "We couldn’t finish every check";
    summary.textContent = "This photo cannot be accepted automatically.";
    continueButton.hidden = true;
  }

  list.replaceChildren(...result.checks.map((check) => {
    const row = document.createElement("div");
    row.className = `check-row${check.passed ? "" : " failed"}`;
    const mark = document.createElement("span");
    mark.className = "check-mark";
    mark.textContent = check.passed ? "✓" : "×";
    const detail = document.createElement("div");
    const label = document.createElement("strong");
    label.textContent = check.label;
    detail.append(label);
    if (!check.passed) {
      const message = document.createElement("p");
      message.textContent = check.message;
      detail.append(message);
    }
    row.append(mark, detail);
    return row;
  }));

  notes.hidden = !result.notes;
  notes.textContent = result.notes || "";
  show(resultView);
}

checkButton.addEventListener("click", async () => {
  if (!selectedFile) return;
  show(loadingView);
  const form = new FormData();
  form.append("file", selectedFile);
  try {
    const response = await fetch("/api/validate", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "The photo could not be checked.");
    renderResult(result);
  } catch (error) {
    renderResult({
      status: "review_required",
      checks: [],
      notes: error.message || "The photo could not be checked. Please try again.",
    });
  }
});

continueButton.addEventListener("click", () => {
  window.dispatchEvent(new CustomEvent("registration-photo-approved"));
  window.alert("Photo accepted. Connect this button to the next registration step.");
});
