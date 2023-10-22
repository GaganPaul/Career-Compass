const popupButton = document.getElementById("popupButton");
const popupForm = document.getElementById("popupForm");
const closeButton = document.getElementById("closeButton");

popupButton.addEventListener("click", () => {
    popupForm.style.display = "block";
});

closeButton.addEventListener("click", () => {
    popupForm.style.display = "none";
});

window.addEventListener("click", (event) => {
    if (event.target === popupForm) {
        popupForm.style.display = "none";
    }
});
