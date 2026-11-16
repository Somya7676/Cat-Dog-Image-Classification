
(function () {
    "use strict";

    window.copyPawSenseResult = async function (label, confidence) {

        const text =
            `PawSense AI Prediction\n` +
            `Result: ${label}\n` +
            `Confidence: ${confidence}`;

        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
            } else {
                const field = document.createElement("textarea");
                field.value = text;
                document.body.appendChild(field);
                field.select();

                const copied = document.execCommand("copy");
                field.remove();

                if (!copied) throw new Error("Copy failed");
            }

            const button = document.getElementById("copyButton");
            const status = document.getElementById("copyStatus");

            if (button) button.textContent = "✓ Copied!";
            if (status) status.textContent = "Prediction copied successfully.";

        } catch (error) {
            const status = document.getElementById("copyStatus");

            if (status) {
                status.textContent = "Copy unavailable. Please copy manually.";
            }
        }
    };

})();
