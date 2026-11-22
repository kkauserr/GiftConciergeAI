const giftRequest = document.getElementById("gift-request");
const searchView = document.getElementById("search-view");
const loadingView = document.getElementById("loading-view");

const loadingSteps = [
    document.getElementById("loading-step-1"),
    document.getElementById("loading-step-2"),
    document.getElementById("loading-step-3"),
    document.getElementById("loading-step-4")
];
const clarificationView = document.getElementById("clarification-view");
const clarificationQuestion = document.getElementById("clarification-question");
const clarificationOptions = document.querySelectorAll(".clarification-option");
const clarificationInput = document.getElementById("clarification-input");
const clarificationSubmit = document.getElementById("clarification-submit");
const clarificationBack = document.getElementById("clarification-back");
const findButton = document.getElementById("find-gifts-button");
const exampleChips = document.querySelectorAll(".example-chip");

const resultsSection = document.getElementById("results-section");
const preferenceSummary = document.getElementById("preference-summary");
const recommendationGrid = document.getElementById("recommendation-grid");

// Conversational Gift Concierge
const conciergeWorkspace = document.getElementById("concierge-workspace");
const conciergeMessages = document.getElementById("concierge-messages");
const conciergeProgress = document.getElementById("concierge-progress");
const conciergeProgressSteps = document.querySelectorAll(".concierge-progress-step");
const conciergeOptions = document.getElementById("concierge-options");
const conciergeInputRow = document.getElementById("concierge-input-row");
const conciergeInput = document.getElementById("concierge-input");
const conciergeSend = document.getElementById("concierge-send");


const examples = [
    "I need a graduation gift for my 22-year-old sister. She loves coffee and reading. Budget is $80. Nothing wearable.",

    "I need a birthday gift for my best friend. She loves puzzles, baking, and cozy nights at home. Keep it under $50.",

    "I'm looking for a thoughtful gift for my dad. He enjoys reading, coffee, and spending time outdoors. Budget is $60."
];


/*
    MOCK BACKEND RESPONSE

    Later, this object will be replaced by the response from:

    POST /recommend

    The UI will not need to be redesigned.
*/

const mockResponse = {

    preferences: [
        { icon: "👤", label: "Sister" },
        { icon: "🎓", label: "Graduation" },
        { icon: "☕", label: "Coffee" },
        { icon: "📚", label: "Reading" },
        { icon: "💰", label: "Under $80" },
        { icon: "⊘", label: "No wearables" }
    ],
recommendations: [
    {
        parent_asin: "1615849858",
        title: "The Professional Barista's Handbook",
        price: 54.90,
        rating: 4.0,
        rating_count: 4,
        category: "Books",
        image: "https://m.media-amazon.com/images/I/51q98+fWDdL._SX350_BO1,204,203,200_.jpg",
        match: 98,
        why_it_fits: "Combines her love of coffee and reading in one thoughtful gift.",
        evidence: "A practical coffee-focused book for someone interested in learning more about professional barista techniques."
    },
    {
        parent_asin: "B07G1XCN93",
        title: "Pour Some Sugar On Me — Coffee Illustration",
        price: 11.85,
        rating: 4.4,
        rating_count: 15,
        category: "Handmade",
        image: "https://m.media-amazon.com/images/I/61jRx+UHg5L._SL1200_.jpg",
        match: 93,
        why_it_fits: "A fun, affordable coffee-themed gift that feels more personal than a generic graduation present.",
        evidence: "Handmade coffee-themed wall art designed for coffee lovers and home or café décor."
    },
    {
        parent_asin: "B0BN9656ZB",
        title: "Coffee Mat for Coffee Maker or Espresso Machine",
        price: 24.00,
        rating: 4.8,
        rating_count: 159,
        category: "Handmade",
        image: "https://m.media-amazon.com/images/I/71p8vXcLNML._SL1500_.jpg",
        match: 86,
        why_it_fits: "A practical addition to a coffee setup that fits comfortably within your budget.",
        evidence: "Rated 4.8/5 from 159 ratings and designed as a washable spill mat for a home coffee station."
    }
]
};


exampleChips.forEach((chip, index) => {

    chip.addEventListener("click", () => {

        giftRequest.value = examples[index];

        giftRequest.focus();

        giftRequest.animate(
            [
                { transform: "scale(1)" },
                { transform: "scale(1.008)" },
                { transform: "scale(1)" }
            ],
            {
                duration: 300,
                easing: "ease-out"
            }
        );

    });

});

function renderPreferences(preferences) {

    preferenceSummary.innerHTML = "";

    const labels = preferences.map(preference => preference.label);

    preferenceSummary.innerHTML = `
        <span class="preference-context-label">Shopping for:</span>
        <span class="preference-context-text">
            ${labels.join(" · ")}
        </span>
    `;
}

function addConciergeMessage(role, text) {
    const message = document.createElement("div");

    message.className = `concierge-message ${role}`;

    const label = role === "user" ? "YOU" : "✦ GIFT CONCIERGE";

    message.innerHTML = `
        <span class="concierge-message-label">${label}</span>
        <div class="concierge-bubble">${text}</div>
    `;

    conciergeMessages.appendChild(message);

    message.scrollIntoView({
        behavior: "smooth",
        block: "nearest"
    });
   return message.querySelector(".concierge-bubble");
}


function showConciergeOptions(options) {
    conciergeOptions.innerHTML = "";

    options.forEach((option) => {
        const button = document.createElement("button");

        button.className = "concierge-option";
        button.textContent = option;

        conciergeOptions.appendChild(button);
    });

    conciergeOptions.classList.remove("hidden");
}

function createProductPlaceholder(category, imageUrl, title) {
    let icon = "🎁";

    if (category === "Books") {
        icon = "📚";
    }

    if (category === "Handmade" || category === "Handmade Products") {
        icon = "✦";
    }

    if (category === "Toys & Games") {
        icon = "🧩";
    }

    if (imageUrl) {
        return `
            <div class="product-image-container">
                <img
                    src="${imageUrl}"
                    alt="${title}"
                    class="product-image"
                    onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';"
                >
                <div class="product-placeholder image-fallback">
                    <span>${icon}</span>
                </div>
            </div>
        `;
    }

    return `
        <div class="product-placeholder">
            <span>${icon}</span>
        </div>
    `;
}

function renderRecommendations(recommendations) {

    recommendationGrid.innerHTML = "";

    recommendations.forEach((product, index) => {

        const card = document.createElement("article");

        card.className = "recommendation-card";

        card.style.animationDelay = `${150 + (index * 110)}ms`;


        const amazonUrl =
            `https://www.amazon.com/dp/${product.parent_asin}`;


        const imageContent = product.image
            ? `<img src="${product.image}" alt="${product.title}">`
            : createProductPlaceholder(product.category);


        card.innerHTML = `

            <div class="product-image">

                ${imageContent}

                <span class="match-badge">
                    ${product.match}
                </span>

            </div>


            <div class="product-content">

                <span class="product-category">
                    ${product.category}
                </span>

                <h4 class="product-title">
                    ${product.title}
                </h4>


                <div class="product-meta">

                    <span class="product-price">
                        $${product.price.toFixed(2)}
                    </span>

                    <span class="product-rating">
                        ★ ${product.rating}
                        · ${product.rating_count} ratings
                    </span>

                </div>


                <div class="why-fit">

                    <span class="why-fit-label">
                        ✦ WHY THIS FITS
                    </span>

                    <p>
                        ${product.why_it_fits}
                    </p>

                    <p class="evidence-line">
                        “${product.evidence}”
                    </p>

                </div>


                <a
                    class="amazon-link"
                    href="${amazonUrl}"
                    target="_blank"
                    rel="noopener noreferrer"
                >
                    View on Amazon
                    <span>→</span>
                </a>

            </div>
        `;


        recommendationGrid.appendChild(card);

    });

}
conciergeProgress.classList.add("progress-finished");

setTimeout(() => {
    conciergeProgress.classList.add("hidden");
    conciergeProgress.classList.remove("progress-finished");
}, 350);
function showResults(data) {

    renderPreferences(data.preferences);
    renderRecommendations(data.recommendations);

    // Results now live inside the conversational workspace.
    resultsSection.classList.remove("hidden");

    // Keep the conversation input available for refinement.
    conciergeInputRow.classList.remove("hidden");

    conciergeInput.placeholder =
        "Ask for something different, change the budget, or refine these gifts...";
}


findButton.addEventListener("click", () => {

    const request = giftRequest.value.trim();

    if (!request) {
        giftRequest.focus();

        giftRequest.animate(
            [
                { transform: "translateX(0)" },
                { transform: "translateX(-5px)" },
                { transform: "translateX(5px)" },
                { transform: "translateX(-3px)" },
                { transform: "translateX(0)" }
            ],
            {
                duration: 320,
                easing: "ease-out"
            }
        );

        return;
    }

    const lowerRequest = request.toLowerCase();

    // Temporary mock clarification logic.
    // Later Shashank's LLM response will determine this.
    const needsClarification =
        lowerRequest.includes("friend") &&
        !request.includes("$") &&
        !lowerRequest.includes("budget");

    // Hide the original search box and examples.
    document.querySelector(".search-card").classList.add("hidden");
    document.querySelector(".example-section").classList.add("hidden");

    // Start the conversational workspace.
    conciergeWorkspace.classList.remove("hidden");
    conciergeMessages.innerHTML = "";
    conciergeOptions.innerHTML = "";

    conciergeOptions.classList.add("hidden");
    conciergeInputRow.classList.add("hidden");
    conciergeProgress.classList.add("hidden");

    // Show what the user originally asked.
    addConciergeMessage("user", request);

    setTimeout(() => {

        if (needsClarification) {

            addConciergeMessage(
                "assistant",
                "I'd love to help. What budget are you working with?"
            );

            showConciergeOptions([
                "Under $25",
                "$25–$50",
                "$50–$100",
                "No limit"
            ]);

            conciergeInputRow.classList.remove("hidden");

        } else {

            addConciergeMessage(
                "assistant",
                "Perfect — I have enough to start looking. Let me find the strongest matches for you."
            );

            conciergeProgress.classList.remove("hidden");

            // Reset progress.
            conciergeProgressSteps.forEach((step, index) => {
                step.classList.remove("active", "complete");

                const icon = step.querySelector(".progress-icon");
                icon.textContent = index + 1;
            });

            // Step 1
            conciergeProgressSteps[0].classList.add("active");

            setTimeout(() => {
                conciergeProgressSteps[0].classList.remove("active");
                conciergeProgressSteps[0].classList.add("complete");
                conciergeProgressSteps[0].querySelector(".progress-icon").textContent = "✓";

                conciergeProgressSteps[1].classList.add("active");
            }, 700);

            // Step 2
            setTimeout(() => {
                conciergeProgressSteps[1].classList.remove("active");
                conciergeProgressSteps[1].classList.add("complete");
                conciergeProgressSteps[1].querySelector(".progress-icon").textContent = "✓";

                conciergeProgressSteps[2].classList.add("active");
            }, 1400);

            // Step 3
            setTimeout(() => {
                conciergeProgressSteps[2].classList.remove("active");
                conciergeProgressSteps[2].classList.add("complete");
                conciergeProgressSteps[2].querySelector(".progress-icon").textContent = "✓";

                conciergeProgressSteps[3].classList.add("active");
            }, 2100);

            // Step 4
            setTimeout(() => {
                conciergeProgressSteps[3].classList.remove("active");
                conciergeProgressSteps[3].classList.add("complete");
                conciergeProgressSteps[3].querySelector(".progress-icon").textContent = "✓";
            }, 2800);

            // Finish
            setTimeout(() => {

               const resultsBubble = addConciergeMessage(
                    "assistant",
                    "I found a few gifts worth considering. These are the strongest matches based on what you told me."
                );
                resultsBubble.classList.add("results-bubble");

                resultsBubble.appendChild(resultsSection);

                setTimeout(() => {
                    showResults(mockResponse);
                }, 650);

            }, 3300);
        }

    }, 450);

});




