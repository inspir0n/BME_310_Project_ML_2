// Shows a preview of the chosen X-ray before the form is submitted.
// This is the only JavaScript on the whole site — everything else is
// plain HTML forms and full-page loads, on purpose, since that's the
// simplest thing that works for a form-and-results page like this one.

const input = document.getElementById("xray-input");
const preview = document.getElementById("preview");
const label = document.getElementById("dropzone-label");

input.addEventListener("change", () => {
  const file = input.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (event) => {
    preview.src = event.target.result;
    preview.style.display = "block";
    label.style.display = "none";
  };
  reader.readAsDataURL(file);
});
