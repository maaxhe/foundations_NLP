/**
 * main.js -- Frontend logic for the D&D NPC Chat UI.
 *
 * Responsibilities:
 *   1. On page load, POST /session/new to create a session and display the
 *      NPC's opening line.
 *   2. On "Send" button click (or Enter key), POST /chat with the player's
 *      message and the current session_id.
 *   3. Render the NPC's response in the chat window.
 *   4. Update the NPC State Panel with data from NPCResponse.current_state.
 *
 * No external JavaScript libraries are required (plain fetch API).
 * TODO: Consider adding a lightweight framework like Alpine.js or htmx
 *       for cleaner DOM management if the UI grows in complexity.
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/**
 * @type {string|null}
 * TODO: Assign this from the /session/new response.
 *       Generate a client-side UUID as a fallback:
 *           sessionId = crypto.randomUUID();
 */
let sessionId = null;

/**
 * @type {string}
 * TODO: Read from a config or let the user choose the NPC from a dropdown.
 */
const npcName = "Grimtooth";


// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

/**
 * POST /session/new and display the opening line.
 *
 * TODO: Implement:
 *   const res = await fetch("/session/new", {
 *       method: "POST",
 *       headers: {"Content-Type": "application/json"},
 *       body: JSON.stringify({npc_name: npcName}),
 *   });
 *   const data = await res.json();
 *   sessionId = data.session_id;
 *   document.getElementById("opening-line").textContent = data.opening_line;
 *   appendMessage("npc", data.opening_line);
 *
 * TODO: Show a loading spinner in the chat window while waiting for the
 *       /session/new response to avoid a blank screen on slow hardware.
 */
async function initSession() {
    // TODO: implement
    console.warn("initSession: not implemented");
}


/**
 * POST /chat with the player's message and render the response.
 *
 * @param {string} playerMessage - Text typed by the player.
 *
 * TODO: Implement:
 *   const res = await fetch("/chat", {
 *       method: "POST",
 *       headers: {"Content-Type": "application/json"},
 *       body: JSON.stringify({
 *           player_message: playerMessage,
 *           session_id: sessionId,
 *           npc_name: npcName,
 *       }),
 *   });
 *   const data = await res.json();
 *   appendMessage("npc", data.npc_message);
 *   updateStatePanel(data.current_state);
 *
 * TODO: Disable the Send button and show a typing indicator while awaiting
 *       the server response, to improve perceived responsiveness.
 * TODO: Handle HTTP error responses (4xx, 5xx) and display a user-friendly
 *       error message in the chat window instead of crashing silently.
 */
async function sendMessage(playerMessage) {
    // TODO: implement
    console.warn("sendMessage: not implemented", playerMessage);
}


// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

/**
 * Append a chat bubble to the chat window.
 *
 * @param {"player"|"npc"} role
 * @param {string} text
 *
 * TODO: Implement:
 *   const chatWindow = document.getElementById("chat-window");
 *   const div = document.createElement("div");
 *   div.className = `turn ${role}`;
 *   div.innerHTML = `<strong>${role === "player" ? "You" : npcName}:</strong> ${escapeHtml(text)}`;
 *   chatWindow.appendChild(div);
 *   chatWindow.scrollTop = chatWindow.scrollHeight;
 *
 * TODO: Add a CSS animation (fade-in) on new turn divs for polish.
 * TODO: Implement escapeHtml to prevent XSS from NPC-generated text.
 */
function appendMessage(role, text) {
    // TODO: implement
    console.warn("appendMessage: not implemented", role, text);
}


/**
 * Update the NPC State Panel with fresh state data.
 *
 * @param {Object} state - The StateSnapshot object from NPCResponse.current_state.
 * @param {string} state.aggression
 * @param {number} state.trust_score
 * @param {string[]} state.inventory_summary
 * @param {string|null} state.active_quest
 * @param {string} state.last_player_intent
 *
 * TODO: Implement:
 *   document.getElementById("state-aggression").textContent = state.aggression;
 *   document.getElementById("state-trust").textContent =
 *       `${(state.trust_score * 100).toFixed(0)}%`;
 *   document.getElementById("state-quest").textContent =
 *       state.active_quest ?? "None";
 *   document.getElementById("state-intent").textContent =
 *       state.last_player_intent || "—";
 *   const ul = document.getElementById("state-inventory");
 *   ul.innerHTML = "";
 *   state.inventory_summary.forEach(item => {
 *       const li = document.createElement("li");
 *       li.textContent = item;
 *       ul.appendChild(li);
 *   });
 *
 * TODO: Add a colour-coded badge for aggression level:
 *       friendly=green, neutral=grey, suspicious=orange, hostile=red.
 *       Use CSS classes toggled via JS.
 */
function updateStatePanel(state) {
    // TODO: implement
    console.warn("updateStatePanel: not implemented", state);
}


// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

/**
 * TODO: Attach event listeners after the DOM is ready.
 *
 *   document.addEventListener("DOMContentLoaded", async () => {
 *       await initSession();
 *
 *       document.getElementById("send-btn").addEventListener("click", async () => {
 *           const input = document.getElementById("player-input");
 *           const text = input.value.trim();
 *           if (!text) return;
 *           input.value = "";
 *           appendMessage("player", text);
 *           await sendMessage(text);
 *       });
 *
 *       document.getElementById("player-input").addEventListener("keydown", (e) => {
 *           if (e.key === "Enter") {
 *               document.getElementById("send-btn").click();
 *           }
 *       });
 *   });
 */

// TODO: remove this stub and implement the DOMContentLoaded block above.
console.log("main.js loaded — waiting for implementation.");
