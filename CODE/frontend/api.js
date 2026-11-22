/*
    Gift Concierge — API / service layer

    The UI calls only:
        GiftConciergeAPI.newSessionId()
        GiftConciergeAPI.sendMessage(sessionId, message)  -> Promise<normalized response>
        GiftConciergeAPI.resetSession(sessionId)          -> Promise<void>

    Where responses come from:
        served by CODE/backend (http://localhost:8000)   real backend, detected via GET /health
        opened as a file, or no backend answering        mock backend (mock-backend.js)
        ?api=<base url>    force the real backend at that URL (POST <base>/recommend, /reset)
        ?api=same-origin   force the real backend on this page's host
        ?mock=1            force the mock
        window.GIFT_CONCIERGE_CONFIG = { apiBase: "...", useMock: false } also works.

    normalizeResponse() turns whatever the backend returns into the one shape
    the UI renders. It accepts:
        1. the UI-facing shape (recommendations: [...]) — preferred
        2. the raw GiftConcierge.handle() shape from CODE/llm/conversation.py
           (explanations + evidence + ranked), so an early backend can return
           handle() output directly.
    The UI never sees backend internals (evidence IDs, retrieval sources, scores).
*/

window.GiftConciergeAPI = (() => {
    "use strict";

    const REQUEST_TIMEOUT_MS = 60000;

    class GiftConciergeError extends Error {
        constructor(kind, message) {
            super(message);
            this.kind = kind;   // "network" | "server" | "malformed"
        }
    }

    const config = { apiBase: "", useMock: true, source: "mock" };

    async function backendAnswers(base) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 1500);
        try {
            const response = await fetch(`${base}/health`, { signal: controller.signal });
            const body = response.ok ? await response.json() : null;
            return Boolean(body && body.service === "gift-concierge");
        } catch {
            return false;
        } finally {
            clearTimeout(timer);
        }
    }

    // Decided once, before the first request.
    const ready = (async () => {
        const explicit = window.GIFT_CONCIERGE_CONFIG || {};
        const params = new URLSearchParams(window.location.search);
        const apiParam = params.get("api");

        if (typeof explicit.useMock === "boolean") {
            Object.assign(config, explicit, { source: "config" });
        } else if (params.get("mock") === "1") {
            Object.assign(config, { useMock: true, source: "?mock=1" });
        } else if (apiParam) {
            Object.assign(config, { useMock: false, apiBase: apiParam === "same-origin" ? "" : apiParam, source: "?api" });
        } else if (/^https?:$/.test(window.location.protocol) && await backendAnswers("")) {
            Object.assign(config, { useMock: false, apiBase: "", source: "detected backend" });
        }

        config.apiBase = String(config.apiBase || "").replace(/\/+$/, "");
        if (window.console) {
            console.info(`[Gift Concierge] using ${config.useMock ? "mock backend" : `backend at ${config.apiBase || window.location.origin}`} (${config.source})`);
        }
    })();


    /* ---------- transport ---------- */

    async function postJSON(path, body) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

        let response;
        try {
            response = await fetch(`${config.apiBase}${path}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: controller.signal
            });
        } catch (error) {
            throw new GiftConciergeError("network", error.name === "AbortError"
                ? "The request timed out."
                : "Could not reach the Gift Concierge service.");
        } finally {
            clearTimeout(timer);
        }

        if (!response.ok) {
            throw new GiftConciergeError("server", `The service responded with ${response.status}.`);
        }

        try {
            return await response.json();
        } catch {
            throw new GiftConciergeError("malformed", "The service returned an unreadable response.");
        }
    }

    async function sendMessage(sessionId, message) {
        await ready;
        let raw;

        if (config.useMock) {
            try {
                raw = await window.MockGiftConcierge.handle(sessionId, message);
            } catch (error) {
                throw new GiftConciergeError(error.kind || "network", error.message);
            }
        } else {
            raw = await postJSON("/recommend", { session_id: sessionId, message });
        }

        return normalizeResponse(raw);
    }

    async function resetSession(sessionId) {
        if (!sessionId) {
            return;
        }

        await ready;
        if (config.useMock) {
            window.MockGiftConcierge.reset(sessionId);
            return;
        }

        try {
            await postJSON("/reset", { session_id: sessionId });
        } catch {
            // A failed reset is harmless: the next conversation uses a new session id.
        }
    }

    function newSessionId() {
        if (window.crypto && typeof window.crypto.randomUUID === "function") {
            return window.crypto.randomUUID();
        }
        return `s-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    }


    /* ---------- normalization ---------- */

    const DEFAULT_RESULTS_MESSAGE =
        "I found a few gifts worth considering. These are the strongest matches based on what you told me.";

    function toNumber(value) {
        const n = typeof value === "string" ? parseFloat(value.replace(/[^0-9.]/g, "")) : value;
        return Number.isFinite(n) ? n : null;
    }

    function cleanText(value, limit = 240) {
        if (typeof value !== "string") {
            return "";
        }
        const text = value.replace(/\s+/g, " ").trim();
        return text.length > limit ? `${text.slice(0, limit - 1).trimEnd()}…` : text;
    }

    function amazonUrlFor(asin) {
        return asin ? `https://www.amazon.com/dp/${encodeURIComponent(asin)}` : null;
    }

    // One human-readable highlight per card. Never an internal ID.
    function normalizeHighlight(item) {
        if (!item) {
            return null;
        }
        if (typeof item === "string") {
            return { text: cleanText(item, 170), source: "listing" };
        }
        const text = cleanText(item.text || item.snippet || "", 170);
        if (!text) {
            return null;
        }
        const source = item.source === "review" || item.kind === "review" ? "review" : (item.source || "listing");
        return { text, source };
    }

    function normalizeRecommendation(rec, index) {
        const asin = rec.parent_asin || rec.asin || null;
        const highlights = []
            .concat(rec.review_evidence || [])
            .concat(rec.highlights || [])
            .concat(rec.evidence ? [rec.evidence] : [])
            .map(normalizeHighlight)
            .filter(Boolean);

        return {
            parent_asin: asin,
            rank: rec.rank || index + 1,
            title: cleanText(rec.title || "Untitled product", 160),
            price: toNumber(rec.price ?? rec.price_num),
            rating: toNumber(rec.rating ?? rec.average_rating),
            rating_count: toNumber(rec.rating_count ?? rec.rating_number),
            category: rec.category || rec.domain || "",
            image_url: rec.image_url || rec.image || null,
            amazon_url: rec.amazon_url || amazonUrlFor(asin),
            why_it_fits: cleanText(rec.why_it_fits || rec.explanation || "", 320),
            highlight: highlights[0] || null
        };
    }

    // Raw GiftConcierge.handle() output: build cards from explanations + evidence lines.
    function recommendationsFromHandle(raw) {
        const evidenceByPid = raw.evidence || {};

        return (raw.explanations || []).map((exp, index) => {
            const pid = Object.keys(evidenceByPid).find(key =>
                (evidenceByPid[key] || []).some(line => line.parent_asin === exp.parent_asin)
            );
            const lines = pid ? evidenceByPid[pid] : [];
            const byField = field => lines.find(line => line.field === field);

            const ratingText = (byField("average_rating") || {}).text || "";
            const ratingMatch = ratingText.match(/([\d.]+)\/5(?: from ([\d,]+) ratings)?/);

            const used = exp.used_evidence || [];
            const review = used.find(line => line.field === "review_samples");
            const kgFact = used.find(line => line.field === "kg");

            let highlight = null;
            if (review) {
                highlight = { text: review.text.replace(/^(\d+(\.\d+)?\/5 review|Review):\s*/, ""), source: "review" };
            } else if (kgFact && !/^Caution:/i.test(kgFact.text)) {
                highlight = { text: kgFact.text, source: "insight" };
            }

            return normalizeRecommendation({
                parent_asin: exp.parent_asin,
                rank: exp.rank,
                title: (byField("title") || {}).text,
                price: (byField("price_num") || {}).text,
                rating: ratingMatch ? ratingMatch[1] : null,
                rating_count: ratingMatch && ratingMatch[2] ? ratingMatch[2] : null,
                category: ((byField("specific_category") || {}).text || "").replace(/^Category:\s*/, ""),
                image_url: exp.image_url || null,
                explanation: exp.explanation,
                highlights: highlight ? [highlight] : []
            }, index);
        });
    }

    function normalizeResponse(raw) {
        if (!raw || typeof raw !== "object" || !["question", "recommendations"].includes(raw.type)) {
            throw new GiftConciergeError("malformed", "The response was missing a valid type.");
        }

        const constraints = raw.constraints && typeof raw.constraints === "object" ? raw.constraints : {};
        const suggestions = Array.isArray(raw.suggestions) ? raw.suggestions.filter(s => typeof s === "string") : [];

        if (raw.type === "question") {
            if (typeof raw.message !== "string" || !raw.message.trim()) {
                throw new GiftConciergeError("malformed", "The question had no text.");
            }
            return {
                type: "question",
                message: raw.message.trim(),
                slot: raw.slot || null,
                constraints,
                suggestions,
                recommendations: []
            };
        }

        const isUiShape = Array.isArray(raw.recommendations);
        const recommendations = isUiShape
            ? raw.recommendations.map(normalizeRecommendation)
            : recommendationsFromHandle(raw);

        // handle() puts a plain-text list in `message`; the cards replace it.
        const message = isUiShape && typeof raw.message === "string" && raw.message.trim()
            ? raw.message.trim()
            : (raw.summary || DEFAULT_RESULTS_MESSAGE);

        return {
            type: "recommendations",
            message,
            followup: typeof raw.followup === "string" ? raw.followup : null,
            slot: raw.slot || null,
            constraints,
            suggestions,
            recommendations
        };
    }

    return {
        config,
        ready,
        sendMessage,
        resetSession,
        newSessionId,
        normalizeResponse,
        GiftConciergeError
    };
})();
