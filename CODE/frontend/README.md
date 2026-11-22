# Gift Concierge frontend

Plain HTML/CSS/JS, no build step. Open `index.html` in a browser (or serve the folder).

## Files

| File | Role |
|---|---|
| `index.html` | Landing screen, conversation workspace, composer, progress template |
| `styles.css` | ClickLessAI visual language (cream, navy, cyan/teal) and conversation layout |
| `app.js` | UI only: session state, message rendering, progress, cards, quick replies, start over |
| `api.js` | The backend boundary: `sendMessage`, `resetSession`, `newSessionId`, `normalizeResponse` |
| `mock-backend.js` | **Temporary** stand-in for the backend. Delete it (and its `<script>` tag) once the API is live |

The UI never decides anything about gifts. Everything it shows comes from the response returned by `api.js`.

## Running

| How you open it | Backend used |
|---|---|
| `http://localhost:8000` (served by `CODE/backend`) | Real backend, detected automatically via `GET /health` |
| `index.html` opened as a file, or no backend answering | Mock backend |
| `…?api=http://localhost:8000` | Force the real backend at that URL |
| `…?mock=1` | Force the mock |

The browser console prints which one is in use. See `CODE/backend/README.md` to start the backend.

Mock-only test hooks: put `#error` in a message to simulate a network failure, or `#malformed` for an unreadable response.

## API contract

### `POST /recommend`

Request:

```json
{ "session_id": "3f6c…", "message": "I need a graduation gift for my sister…" }
```

`session_id` is created by the frontend (one per conversation; a new one after **Start over**). The backend keeps one `GiftConcierge` / `DialogueState` per session id.

Response, clarifying question:

```json
{
  "type": "question",
  "message": "What is your friend into? A couple of hobbies or interests will help me find something that fits.",
  "slot": "interests",
  "turn": 1,
  "constraints": { "...": "to_handoff() output" },
  "suggestions": []
}
```

Response, recommendations (preferred UI-facing shape):

```json
{
  "type": "recommendations",
  "message": "I found a few gifts worth considering.",
  "turn": 3,
  "constraints": { "recipient": "sister", "occasion": "graduation", "interests": ["coffee", "reading"],
                   "budget_max": 80, "exclusions_stated": ["wearable"] },
  "recommendations": [
    {
      "parent_asin": "B012345",
      "rank": 1,
      "title": "Example Product",
      "price": 49.99,
      "rating": 4.7,
      "rating_count": 1234,
      "category": "Handmade Products",
      "image_url": "https://…",
      "amazon_url": "https://www.amazon.com/dp/B012345",
      "why_it_fits": "A concise grounded explanation…",
      "review_evidence": [{ "text": "My sister uses it every morning.", "source": "review" }]
    }
  ],
  "suggestions": ["More unique", "A little cheaper"]
}
```

Notes:

- `suggestions` is optional. Without it, the UI offers default refinements after results, and budget ranges when `slot` is `"budget"`.
- `review_evidence` is optional. Only the first item is shown, as one short line on the card. `source` is `"review"` (shown as a quote), `"insight"`, `"ratings"` or `"listing"`. Never send evidence IDs.
- "Shopping for" is built from `constraints`: `recipient`, `occasion`, `interests`, `budget_min` / `budget_max` / `budget_flexible`, and `exclusions_stated` (falls back to `exclusions`).
- An early backend can also return the raw `GiftConcierge.handle()` dict unchanged (`explanations` + `evidence`). `normalizeResponse()` rebuilds title, price, rating and category from the evidence lines and uses a cited review as the highlight. Images need the UI shape (`image_url`).
- Anything else (no `type`, empty question) is shown as a friendly error in the conversation.

### `POST /reset`

Request: `{ "session_id": "3f6c…" }`. Called on **Start over**. Optional: if it fails, the next conversation uses a new session id anyway.

## Backend

`CODE/backend` implements this contract on top of `GiftConcierge.handle()`: one conversation per `session_id`, product details from the corpus, `image_url` from the `build_image_lookup.py` output, and `amazon_url` from `parent_asin`.
