/*
    MOCK BACKEND — FOR FRONTEND DEVELOPMENT ONLY. DELETE WHEN THE REAL API IS LIVE.

    Stands in for POST /recommend so the conversation UI can be exercised
    without the LLM / retrieval / ranking stack. It is deliberately crude:
    regex "extraction", a 13-product fixture list, and keyword matching.
    None of this logic belongs in the real frontend; the real decisions are
    made by CODE/llm (constraint extraction, dialogue state, clarification)
    and CODE/ranking (retrieval, filtering, ranking).

    It returns the UI-facing response shape documented in README.md, and it
    copies the real backend's clarification wording and its "cheaper" rule
    (budget × 0.75) so the flows look like the real thing.

    Test hooks (mock only): include "#error" in a message to simulate a
    network failure, "#malformed" for an unreadable response.

    Products are real items from DATA/samples/sample_corpus_v2.jsonl
    (plus the three coffee items from the original mock). Explanations are
    placeholder copy, not model output.
*/

window.MockGiftConcierge = (() => {
    "use strict";

    const PRODUCTS = [
        { parent_asin: "1615849858", title: "The Professional Barista's Handbook", price: 54.90, rating: 4.0, rating_count: 4,
          category: "Books", image_url: "https://m.media-amazon.com/images/I/51q98+fWDdL._SX350_BO1,204,203,200_.jpg",
          tags: ["coffee", "reading"], practical: true,
          why: "Brings coffee and reading together: a hands-on guide to professional espresso technique.",
          highlight: { text: "A practical guide to professional barista techniques.", source: "listing" } },
        { parent_asin: "B07G1XCN93", title: "Pour Some Sugar On Me — Coffee Illustration", price: 11.85, rating: 4.4, rating_count: 15,
          category: "Handmade", image_url: "https://m.media-amazon.com/images/I/61jRx+UHg5L._SL1200_.jpg",
          tags: ["coffee"], unique: true,
          why: "A playful, handmade coffee print that feels more personal than a generic gift.",
          highlight: { text: "Handmade coffee-themed wall art for a home or café corner.", source: "listing" } },
        { parent_asin: "B0BN9656ZB", title: "Coffee Mat for Coffee Maker or Espresso Machine", price: 24.00, rating: 4.8, rating_count: 159,
          category: "Handmade", image_url: "https://m.media-amazon.com/images/I/71p8vXcLNML._SL1500_.jpg",
          tags: ["coffee"], practical: true,
          why: "A useful upgrade for a home coffee station that fits comfortably in the budget.",
          highlight: { text: "Rated 4.8 out of 5 across 159 ratings.", source: "ratings" } },
        { parent_asin: "B0754TWHPT", title: "Handmade Coffee Stud Earrings", price: 12.95, rating: 4.6, rating_count: 35,
          category: "Handmade", image_url: null,
          tags: ["coffee", "wearable"], unique: true,
          why: "Tiny handmade coffee-cup earrings, a fun everyday nod to her favorite drink.",
          highlight: { text: "Handmade, coffee-cup design.", source: "listing" } },
        { parent_asin: "B08NFD6RHQ", title: "Ravensburger Eibsee Lake, Zugspitze 1000-Piece Jigsaw Puzzle", price: 29.22, rating: 4.7, rating_count: 866,
          category: "Toys & Games", image_url: null,
          tags: ["puzzles"],
          why: "A scenic 1000-piece puzzle for cozy nights in, from a well-loved puzzle brand.",
          highlight: { text: "Rated 4.7 out of 5 across 866 ratings.", source: "ratings" } },
        { parent_asin: "1771513160", title: "Bisous and Brioche: Classic French Recipes and Family Favorites", price: 44.06, rating: 4.7, rating_count: 100,
          category: "Books", image_url: null,
          tags: ["cooking", "baking", "reading"], unique: true,
          why: "A beautiful French cookbook for someone who loves spending time in the kitchen.",
          highlight: { text: "Classic French recipes and family favorites.", source: "listing" } },
        { parent_asin: "0696002655", title: "Better Homes and Gardens All-Time Favorite Recipes", price: 12.15, rating: 4.9, rating_count: 33,
          category: "Books", image_url: null,
          tags: ["cooking", "baking", "reading"], practical: true,
          why: "Dependable, well-reviewed recipes for everyday cooking and baking.",
          highlight: { text: "Rated 4.9 out of 5 across 33 ratings.", source: "ratings" } },
        { parent_asin: "B07M62GVLN", title: "Microwavable Hot Bowl Cozy Holder, Reversible 100% Cotton", price: 13.99, rating: 4.8, rating_count: 133,
          category: "Handmade", image_url: null,
          tags: ["cooking"], practical: true, unique: true,
          why: "A handmade kitchen helper that is genuinely useful every day.",
          highlight: { text: "Reversible, 100% cotton, microwave-safe.", source: "listing" } },
        { parent_asin: "1478235845", title: "The Longest Trail, Vol. 1", price: 44.00, rating: 4.6, rating_count: 59,
          category: "Books", image_url: null,
          tags: ["outdoors", "reading"],
          why: "Trail stories for someone who loves being outside.",
          highlight: { text: "Rated 4.6 out of 5 across 59 ratings.", source: "ratings" } },
        { parent_asin: "B002KE4884", title: "The Annotated Secret Garden", price: 14.95, rating: 4.7, rating_count: 74,
          category: "Books", image_url: null,
          tags: ["gardening", "reading"], unique: true,
          why: "A richly annotated edition of a classic, ideal for a reader who gardens.",
          highlight: { text: "Annotated edition of the classic novel.", source: "listing" } },
        { parent_asin: "B07RHC3YP5", title: "Purple Glass Iris Flower Figurine", price: 17.00, rating: 4.9, rating_count: 37,
          category: "Handmade", image_url: null,
          tags: ["gardening", "decor"], unique: true,
          why: "A hand-blown glass flower that brings the garden indoors.",
          highlight: { text: "Hand-blown glass.", source: "listing" } },
        { parent_asin: "1402766238", title: "Woodstock: Three Days That Rocked the World", price: 35.81, rating: 4.8, rating_count: 246,
          category: "Books", image_url: null,
          tags: ["music", "reading"],
          why: "A photo-rich history of Woodstock for a music lover.",
          highlight: { text: "Rated 4.8 out of 5 across 246 ratings.", source: "ratings" } },
        { parent_asin: "006240749X", title: "The Good Liar: A Novel", price: 14.64, rating: 4.0, rating_count: 3535,
          category: "Books", image_url: null,
          tags: ["reading"],
          why: "A twisty, popular novel for someone who always has a book on the go.",
          highlight: { text: "Over 3,500 ratings.", source: "ratings" } }
    ];

    const INTERESTS = {
        "coffee": /\b(coffee|espresso|latte|caffeine)\b/,
        "reading": /\b(read(ing|er|s)?|books?|novels?|bookworm)\b/,
        "puzzles": /\b(puzzles?|jigsaw)\b/,
        "baking": /\b(bak(e|ing|er)|pastry)\b/,
        "cooking": /\b(cook(ing)?|chef|recipes?|kitchen)\b/,
        "Formula 1": /\b(formula ?(1|one)|f1|racing)\b/,
        "outdoors": /\b(outdoors?|hik(e|ing)|camping|trails?)\b/,
        "gardening": /\b(garden(ing|er)?|plants?)\b/,
        "music": /\b(music|guitar|concerts?)\b/
    };
    const RECIPIENTS = ["sister", "brother", "mom", "dad", "mother", "father", "friend", "wife", "husband",
        "girlfriend", "boyfriend", "nephew", "niece", "aunt", "uncle", "coworker", "boss", "grandma", "grandpa"];
    const OCCASIONS = ["graduation", "birthday", "christmas", "anniversary", "wedding", "holiday", "housewarming", "retirement"];

    const sessions = new Map();

    function newState() {
        return { recipient: null, occasion: null, interests: [], exclusions: [],
                 budget_min: null, budget_max: null, budget_flexible: false,
                 asked: new Set(), turn: 0, lastShown: [] };
    }

    function wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    function money(n) {
        return Math.round(n * 100) / 100;
    }

    function applyMessage(state, text) {
        const t = text.toLowerCase();
        const mods = { unique: /\bunique|different|original|special\b/.test(t),
                       practical: /\bpractical|useful\b/.test(t),
                       similar: /\b(similar|like (this|that|the (first|second|third)))\b/.test(t) };

        RECIPIENTS.forEach(r => { if (new RegExp(`\\b${r}\\b`).test(t)) state.recipient = r; });
        OCCASIONS.forEach(o => { if (t.includes(o)) state.occasion = o; });

        // "not into X anymore" → remove + exclude (mirrors the backend rule)
        Object.entries(INTERESTS).forEach(([name, re]) => {
            const negated = new RegExp(`not (into|interested in|a fan of)[^.]*?${re.source}|${re.source}[^.]*?anymore`).test(t)
                || new RegExp(`\\bno(thing)? ${re.source}`).test(t);
            if (negated) {
                state.interests = state.interests.filter(i => i !== name);
                if (!state.exclusions.includes(name)) state.exclusions.push(name);
            } else if (re.test(t) && !state.interests.includes(name)) {
                state.interests.push(name);
                state.exclusions = state.exclusions.filter(e => e !== name);
            }
        });
        if (/\b(nothing|no) (wearable|clothes|clothing|jewelry)s?\b/.test(t) && !state.exclusions.includes("wearable")) {
            state.exclusions.push("wearable");
        }
        if (/\b(nothing|no) decor(ative)?\b/.test(t) && !state.exclusions.includes("decor")) {
            state.exclusions.push("decor");
        }

        const range = t.match(/\$?\s?(\d+)\s?(?:-|–|to)\s?\$?\s?(\d+)/);
        const amount = t.match(/\$\s?(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s?(?:dollars|bucks)|(?:under|below|budget (?:is|of|to)|around|about|max(?:imum)?)\s\$?(\d+)/);
        let cheaperWithoutBudget = false;

        if (/\bno (limit|budget)|budget doesn'?t matter|any budget\b/.test(t)) {
            Object.assign(state, { budget_min: null, budget_max: null, budget_flexible: true });
        } else if (range) {
            Object.assign(state, { budget_min: +range[1], budget_max: +range[2], budget_flexible: false });
        } else if (amount) {
            Object.assign(state, { budget_min: null, budget_max: +(amount[1] || amount[2] || amount[3]), budget_flexible: false });
        } else if (/\b(cheaper|lower budget|less expensive|lower price)\b/.test(t)) {
            if (state.budget_max) {
                state.budget_max = money(state.budget_max * 0.75);
            } else {
                cheaperWithoutBudget = true;
            }
        }

        return { mods, cheaperWithoutBudget };
    }

    function constraintsOf(state) {
        return {
            recipient: state.recipient,
            occasion: state.occasion,
            interests: [...state.interests],
            budget_min: state.budget_min,
            budget_max: state.budget_max,
            budget_flexible: state.budget_flexible,
            exclusions: [...state.exclusions],
            exclusions_stated: [...state.exclusions]
        };
    }

    function question(state, message, slot, suggestions = []) {
        return { type: "question", message, slot, turn: state.turn, constraints: constraintsOf(state), suggestions };
    }

    function pick(state, mods, lastMessage) {
        let pool = PRODUCTS.filter(p =>
            p.tags.some(tag => state.interests.includes(tag)) &&
            !p.tags.some(tag => state.exclusions.includes(tag)) &&
            (state.budget_max == null || p.price <= state.budget_max) &&
            (state.budget_min == null || p.price >= state.budget_min)
        );

        const score = p => {
            let s = p.tags.filter(tag => state.interests.includes(tag)).length * 2 + p.rating / 5;
            if (mods.unique && p.unique) s += 3;
            if (mods.practical && p.practical) s += 3;
            return s;
        };

        if (mods.similar && state.lastShown.length) {
            const ordinal = /second/.test(lastMessage) ? 1 : /third/.test(lastMessage) ? 2 : 0;
            const liked = PRODUCTS.find(p => p.parent_asin === state.lastShown[ordinal]);
            if (liked) {
                pool = pool.filter(p => p.parent_asin !== liked.parent_asin);
                pool.sort((a, b) => (b.category === liked.category) - (a.category === liked.category) || score(b) - score(a));
                return pool.slice(0, 3);
            }
        }

        return pool.sort((a, b) => score(b) - score(a)).slice(0, 3);
    }

    async function handle(sessionId, message) {
        if (/#error/i.test(message)) {
            await wait(900);
            const error = new Error("Simulated network failure (mock).");
            error.kind = "network";
            throw error;
        }
        if (/#malformed/i.test(message)) {
            await wait(700);
            return { unexpected: true };
        }

        if (!sessions.has(sessionId)) sessions.set(sessionId, newState());
        const state = sessions.get(sessionId);
        state.turn += 1;

        const { mods, cheaperWithoutBudget } = applyMessage(state, message);
        const who = state.recipient ? `your ${state.recipient}` : null;

        if (cheaperWithoutBudget) {
            await wait(700);
            return question(state, "Happy to go cheaper. What price range should I stay under?", "budget",
                ["Under $25", "$25–$50", "$50–$100", "No limit"]);
        }
        if (!state.interests.length && !state.asked.has("interests")) {
            state.asked.add("interests");
            await wait(800);
            return question(state, who
                ? `What is ${who} into? A couple of hobbies or interests will help me find something that fits.`
                : "Who is the gift for, and what are they into? A couple of hobbies or interests will help me find something that fits.",
                "interests");
        }
        if (state.budget_max == null && state.budget_min == null && !state.budget_flexible && !state.asked.has("budget")) {
            state.asked.add("budget");
            await wait(800);
            return question(state, `What's your budget for ${who ? `the gift for ${who}` : "this gift"}? A range like $20-$50 is fine, or say 'no limit'.`,
                "budget", ["Under $25", "$25–$50", "$50–$100", "No limit"]);
        }

        await wait(2600 + Math.random() * 900);
        const picks = pick(state, mods, message.toLowerCase());

        if (!picks.length) {
            return question(state, "I couldn't find anything that fits all of those constraints. Could you raise the budget or loosen one of the requirements?",
                "no_results");
        }

        const isUpdate = state.lastShown.length > 0;
        state.lastShown = picks.map(p => p.parent_asin);
        return {
            type: "recommendations",
            message: isUpdate
                ? "Here's an updated set based on what you just told me."
                : "I found a few gifts worth considering. These are the strongest matches based on what you told me.",
            turn: state.turn,
            constraints: constraintsOf(state),
            recommendations: picks.map((p, i) => ({
                parent_asin: p.parent_asin,
                rank: i + 1,
                title: p.title,
                price: p.price,
                rating: p.rating,
                rating_count: p.rating_count,
                category: p.category,
                image_url: p.image_url,
                amazon_url: `https://www.amazon.com/dp/${p.parent_asin}`,
                why_it_fits: p.why,
                review_evidence: [p.highlight]
            }))
        };
    }

    function reset(sessionId) {
        sessions.delete(sessionId);
    }

    return { handle, reset };
})();
