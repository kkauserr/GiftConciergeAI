/*
    Gift Concierge — UI layer

    Responsibilities: collect messages, keep one conversation on screen,
    show progress while a request is in flight, and render whatever the
    API layer returns (questions or recommendations). No recommendation
    logic lives here; see api.js for the backend boundary.

    Sections:
      1. DOM + copy
      2. Session state
      3. Small helpers
      4. Message rendering
      5. Progress (per request, removed when the response arrives)
      6. Assistant responses + recommendation cards
      7. Quick replies
      8. Conversation flow (send, start, start over)
      9. Event wiring
*/

(() => {
    "use strict";

    const API = window.GiftConciergeAPI;


    /* ---------- 1. DOM + copy ---------- */

    const $ = id => document.getElementById(id);

    const dom = {
        body: document.body,
        landing: $("search-view"),
        searchForm: $("search-form"),
        giftRequest: $("gift-request"),
        exampleChips: Array.from(document.querySelectorAll(".example-chip")),
        workspace: $("concierge-workspace"),
        messages: $("concierge-messages"),
        composer: $("concierge-composer"),
        input: $("concierge-input"),
        send: $("concierge-send"),
        startOver: $("start-over-button"),
        progressTemplate: $("progress-template")
    };

    const EXAMPLES = [
        "I need a graduation gift for my 22-year-old sister. She loves coffee and reading. Budget is $80. Nothing wearable.",
        "I need a birthday gift for my best friend. She loves puzzles, baking, and cozy nights at home. Keep it under $50.",
        "I'm looking for a thoughtful gift for my dad. He enjoys reading, coffee, and spending time outdoors. Budget is $60."
    ];

    const RESULTS_FOLLOWUP = "Want me to make these more unique, more practical, or a little cheaper?";
    const RESULTS_QUICK_REPLIES = ["More unique", "A little cheaper", "More practical", "More like the top pick"];
    const BUDGET_QUICK_REPLIES = ["Under $25", "$25–$50", "$50–$100", "No limit"];
    const EMPTY_RESULTS_MESSAGE =
        "I couldn't find gifts that match all of that yet. Could you raise the budget or loosen one of the requirements?";

    const ERROR_COPY = {
        network: "I couldn't reach the recommendation service just now. Check your connection and try again.",
        server: "Something went wrong on my side while looking for gifts. Please try again in a moment.",
        malformed: "I got an incomplete answer back and couldn't show it. Please try again."
    };

    const CATEGORY_LABELS = { "Handmade Products": "Handmade" };
    const CATEGORY_ICONS = { "Books": "📚", "Handmade": "✦", "Handmade Products": "✦", "Toys & Games": "🧩" };
    const HIGHLIGHT_LABELS = {
        review: "From a customer review",
        insight: "Good to know",
        ratings: "Ratings",
        listing: "From the product listing"
    };

    const PROGRESS_STEP_MS = 1400;
    const PROGRESS_MIN_VISIBLE_MS = 700;
    const PROGRESS_FADE_MS = 360;

    const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;


    /* ---------- 2. Session state ---------- */

    const session = {
        id: null,              // backend session id; new one per conversation
        started: false,        // landing has turned into the conversation
        pending: false,        // a request is in flight
        lastUserMessage: null, // for "Try again"
        requestSeq: 0          // bumped on every request and on start over; stale responses are dropped
    };


    /* ---------- 3. Small helpers ---------- */

    function el(tag, className, text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text != null) node.textContent = text;
        return node;
    }

    function wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function scrollToMessage(node, block = "nearest") {
        node.scrollIntoView({ behavior: prefersReducedMotion ? "auto" : "smooth", block });
    }

    function formatMoney(value) {
        return Number.isInteger(value) ? `$${value}` : `$${value.toFixed(2)}`;
    }

    function capitalize(text) {
        return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
    }

    function autosize(textarea) {
        textarea.style.height = "auto";
        textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
    }

    function shake(node) {
        node.animate(
            [
                { transform: "translateX(0)" },
                { transform: "translateX(-5px)" },
                { transform: "translateX(5px)" },
                { transform: "translateX(-3px)" },
                { transform: "translateX(0)" }
            ],
            { duration: 320, easing: "ease-out" }
        );
    }


    /* ---------- 4. Message rendering ---------- */

    function appendMessage(role) {
        const root = el("div", `concierge-message ${role}`);
        root.append(el("span", "concierge-message-label", role === "user" ? "YOU" : "✦ GIFT CONCIERGE"));
        dom.messages.append(root);
        return root;
    }

    function appendUserMessage(text) {
        const root = appendMessage("user");
        root.append(el("div", "concierge-bubble", text));
        scrollToMessage(root);
        return root;
    }


    /* ---------- 5. Progress ---------- */

    // One progress block per request, created inside the pending assistant turn.
    // It is removed from the DOM when the response arrives, so it can never
    // reappear under later content. The next request creates a fresh one.
    function createProgress(container) {
        const node = dom.progressTemplate.content.firstElementChild.cloneNode(true);
        const steps = Array.from(node.querySelectorAll(".concierge-progress-step"));
        const timers = [];
        const startedAt = performance.now();
        let current = 0;

        function setStep(index) {
            current = index;
            steps.forEach((step, i) => {
                step.classList.toggle("complete", i < index);
                step.classList.toggle("active", i === index);
                step.querySelector(".progress-icon").textContent = i < index ? "✓" : String(i + 1);
            });
        }

        setStep(0);
        for (let i = 1; i < steps.length; i += 1) {
            timers.push(setTimeout(() => setStep(i), PROGRESS_STEP_MS * i));
        }
        container.append(node);

        return {
            // Recommendations: tick off the remaining steps, then fade out and remove.
            // Questions / errors: just fade out and remove.
            async finish(outcome) {
                timers.forEach(clearTimeout);

                const elapsed = performance.now() - startedAt;
                if (elapsed < PROGRESS_MIN_VISIBLE_MS) {
                    await wait(PROGRESS_MIN_VISIBLE_MS - elapsed);
                }

                if (outcome === "recommendations") {
                    for (let i = current + 1; i <= steps.length; i += 1) {
                        setStep(i);
                        await wait(prefersReducedMotion ? 0 : 140);
                    }
                }

                node.classList.add("progress-finished");
                await wait(PROGRESS_FADE_MS);
                node.remove();
            },

            cancel() {
                timers.forEach(clearTimeout);
                node.remove();
            }
        };
    }


    /* ---------- 6. Assistant responses + recommendation cards ---------- */

    function preferenceLabels(constraints) {
        const labels = [];
        const c = constraints || {};

        if (c.recipient) labels.push(capitalize(c.recipient));
        if (c.occasion) labels.push(capitalize(c.occasion));
        (c.interests || []).forEach(interest => labels.push(capitalize(String(interest))));

        const min = typeof c.budget_min === "number" ? c.budget_min : null;
        const max = typeof c.budget_max === "number" ? c.budget_max : null;
        if (min != null && max != null) labels.push(`${formatMoney(min)}–${formatMoney(max)}`);
        else if (max != null) labels.push(`Under ${formatMoney(max)}`);
        else if (min != null) labels.push(`${formatMoney(min)}+`);
        else if (c.budget_flexible) labels.push("Any budget");

        const stated = c.exclusions_stated && c.exclusions_stated.length ? c.exclusions_stated : (c.exclusions || []);
        stated.forEach(term => {
            const word = String(term).trim().replace(/^(no|not|nothing|without)\s+/i, "");
            if (!word) return;
            const plural = /^(wearable|book|toy|game|mug|candle|puzzle)$/i.test(word) ? `${word}s` : word;
            labels.push(`No ${plural.toLowerCase()}`);
        });

        return labels;
    }

    function renderPreferenceContext(constraints) {
        const labels = preferenceLabels(constraints);
        if (!labels.length) return null;

        const summary = el("div", "preference-summary");
        summary.append(
            el("span", "preference-context-label", "Shopping for:"),
            el("span", "preference-context-text", labels.join(" · "))
        );
        return summary;
    }

    function createPlaceholder(category) {
        const placeholder = el("div", "product-placeholder");
        placeholder.append(el("span", null, CATEGORY_ICONS[category] || "🎁"));
        placeholder.setAttribute("aria-hidden", "true");
        return placeholder;
    }

    function createProductMedia(rec, isTopPick) {
        const media = el("div", "product-image");

        if (rec.image_url) {
            const img = document.createElement("img");
            img.src = rec.image_url;
            img.alt = rec.title;
            img.loading = "lazy";
            img.referrerPolicy = "no-referrer";
            img.addEventListener("error", () => img.replaceWith(createPlaceholder(rec.category)), { once: true });
            media.append(img);
        } else {
            media.append(createPlaceholder(rec.category));
        }

        if (isTopPick) {
            media.append(el("span", "match-badge", "Top pick"));
        }
        return media;
    }

    function createHighlight(highlight) {
        const line = el("p", "evidence-line");
        const isQuote = highlight.source === "review";
        if (isQuote) line.classList.add("is-quote");

        line.append(
            el("span", "evidence-label", HIGHLIGHT_LABELS[highlight.source] || HIGHLIGHT_LABELS.listing),
            el("span", "evidence-text", isQuote ? `“${highlight.text}”` : highlight.text)
        );
        return line;
    }

    function createRecommendationCard(rec, index) {
        const card = el("article", "recommendation-card");
        card.style.animationDelay = `${120 + index * 110}ms`;

        card.append(createProductMedia(rec, index === 0));

        const content = el("div", "product-content");

        if (rec.category) {
            content.append(el("span", "product-category", CATEGORY_LABELS[rec.category] || rec.category));
        }

        const title = el("h4", "product-title", rec.title);
        title.title = rec.title;
        content.append(title);

        const meta = el("div", "product-meta");
        if (rec.price != null) {
            meta.append(el("span", "product-price", `$${rec.price.toFixed(2)}`));
        }
        if (rec.rating != null) {
            const count = rec.rating_count != null ? ` · ${rec.rating_count.toLocaleString()} ratings` : "";
            meta.append(el("span", "product-rating", `★ ${rec.rating.toFixed(1)}${count}`));
        }
        if (meta.childElementCount) content.append(meta);

        if (rec.why_it_fits || rec.highlight) {
            const why = el("div", "why-fit");
            why.append(el("span", "why-fit-label", "✦ WHY THIS FITS"));
            if (rec.why_it_fits) why.append(el("p", null, rec.why_it_fits));
            if (rec.highlight) why.append(createHighlight(rec.highlight));
            content.append(why);
        }

        if (rec.amazon_url) {
            const link = el("a", "amazon-link");
            link.href = rec.amazon_url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.setAttribute("aria-label", `View ${rec.title} on Amazon (opens in a new tab)`);
            link.append("View on Amazon", el("span", null, "→"));
            content.append(link);
        }

        card.append(content);
        return card;
    }

    function renderRecommendations(recommendations) {
        const grid = el("div", "recommendation-grid");
        recommendations.forEach((rec, index) => grid.append(createRecommendationCard(rec, index)));
        return grid;
    }

    function renderAssistantResponse(root, response) {
        const hasResults = response.type === "recommendations" && response.recommendations.length > 0;
        root.classList.toggle("has-results", hasResults);

        const bubble = el("div", "concierge-bubble");
        root.append(bubble);

        if (response.type === "recommendations" && !hasResults) {
            bubble.append(el("p", "assistant-text", EMPTY_RESULTS_MESSAGE));
            return;
        }

        bubble.append(el("p", "assistant-text", response.message));

        if (hasResults) {
            const results = el("div", "results-section");
            const context = renderPreferenceContext(response.constraints);
            if (context) results.append(context);
            results.append(renderRecommendations(response.recommendations));
            bubble.append(results);

            bubble.append(el("p", "assistant-followup", response.followup || RESULTS_FOLLOWUP));
            bubble.append(renderQuickReplies(response.suggestions.length ? response.suggestions : RESULTS_QUICK_REPLIES));
            return;
        }

        // Clarifying question: optional suggested answers under the bubble.
        const suggestions = response.suggestions.length
            ? response.suggestions
            : (response.slot === "budget" ? BUDGET_QUICK_REPLIES : []);
        if (suggestions.length) {
            root.append(renderQuickReplies(suggestions));
        }
    }

    function renderError(root, error) {
        root.classList.add("is-error");
        const bubble = el("div", "concierge-bubble");
        bubble.append(el("p", "assistant-text", ERROR_COPY[error && error.kind] || ERROR_COPY.server));
        root.append(bubble);

        if (session.lastUserMessage) {
            const retry = renderQuickReplies(["Try again"], () => sendMessage(session.lastUserMessage, { echo: false }));
            root.append(retry);
        }

        if (error && window.console) {
            console.warn("[Gift Concierge]", error.kind || "error", error.message);
        }
    }


    /* ---------- 7. Quick replies ---------- */

    // Suggested responses. Clicking one sends it as a normal message.
    function renderQuickReplies(options, onPick) {
        const row = el("div", "quick-replies");
        options.forEach(option => {
            const button = el("button", "concierge-option", option);
            button.type = "button";
            button.addEventListener("click", () => (onPick ? onPick(option) : sendMessage(option)));
            row.append(button);
        });
        return row;
    }

    // Only the latest assistant turn offers suggestions.
    function removeQuickReplies() {
        dom.messages.querySelectorAll(".quick-replies").forEach(row => row.remove());
    }


    /* ---------- 8. Conversation flow ---------- */

    function setPending(isPending) {
        session.pending = isPending;
        dom.send.disabled = isPending;
        dom.composer.classList.toggle("is-busy", isPending);
    }

    function startConversation() {
        session.started = true;
        session.id = API.newSessionId();

        dom.landing.classList.add("is-leaving");
        dom.startOver.classList.remove("hidden");

        setTimeout(() => {
            if (!session.started) return;   // start over happened during the transition
            dom.body.classList.add("in-conversation");
            dom.workspace.classList.remove("hidden");
            window.scrollTo({ top: 0, behavior: "auto" });
            dom.input.focus({ preventScroll: true });
        }, prefersReducedMotion ? 0 : 280);
    }

    async function sendMessage(text, { echo = true } = {}) {
        const message = String(text || "").trim();
        if (!message || session.pending) return false;

        if (!session.started) startConversation();

        removeQuickReplies();
        if (echo) appendUserMessage(message);
        session.lastUserMessage = message;

        setPending(true);
        const seq = ++session.requestSeq;

        const root = appendMessage("assistant");
        root.classList.add("is-pending");
        const progress = createProgress(root);
        scrollToMessage(root);

        let response = null;
        let error = null;
        try {
            response = await API.sendMessage(session.id, message);
        } catch (caught) {
            error = caught;
        }

        if (seq !== session.requestSeq) {   // conversation was reset while waiting
            progress.cancel();
            root.remove();
            return false;
        }

        await progress.finish(response ? response.type : "error");
        if (seq !== session.requestSeq) {
            root.remove();
            return false;
        }

        root.classList.remove("is-pending");
        if (error) renderError(root, error);
        else renderAssistantResponse(root, response);

        setPending(false);
        scrollToMessage(root, root.classList.contains("has-results") ? "start" : "nearest");
        dom.input.focus({ preventScroll: true });
        return true;
    }

    function startOver() {
        const oldSessionId = session.id;

        session.requestSeq += 1;            // drops any response still in flight
        session.id = null;
        session.started = false;
        session.lastUserMessage = null;
        setPending(false);
        API.resetSession(oldSessionId);

        dom.messages.replaceChildren();
        dom.input.value = "";
        autosize(dom.input);

        dom.body.classList.remove("in-conversation");
        dom.workspace.classList.add("hidden");
        dom.startOver.classList.add("hidden");
        dom.landing.classList.remove("is-leaving");

        dom.giftRequest.value = "";
        window.scrollTo({ top: 0, behavior: "auto" });
        dom.giftRequest.focus({ preventScroll: true });
    }


    /* ---------- 9. Event wiring ---------- */

    function submitOnEnter(event, form) {
        if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            form.requestSubmit();
        }
    }

    dom.exampleChips.forEach((chip, index) => {
        chip.addEventListener("click", () => {
            dom.giftRequest.value = EXAMPLES[index];
            dom.giftRequest.focus();
            dom.giftRequest.animate(
                [{ transform: "scale(1)" }, { transform: "scale(1.008)" }, { transform: "scale(1)" }],
                { duration: 300, easing: "ease-out" }
            );
        });
    });

    dom.searchForm.addEventListener("submit", event => {
        event.preventDefault();
        const request = dom.giftRequest.value.trim();
        if (!request) {
            dom.giftRequest.focus();
            shake(dom.giftRequest);
            return;
        }
        sendMessage(request);
    });

    dom.giftRequest.addEventListener("keydown", event => submitOnEnter(event, dom.searchForm));

    dom.composer.addEventListener("submit", event => {
        event.preventDefault();
        if (session.pending) return;

        const text = dom.input.value.trim();
        if (!text) {
            shake(dom.composer);
            return;
        }

        dom.input.value = "";
        autosize(dom.input);
        sendMessage(text);
    });

    dom.input.addEventListener("keydown", event => submitOnEnter(event, dom.composer));
    dom.input.addEventListener("input", () => autosize(dom.input));

    dom.startOver.addEventListener("click", startOver);
})();
