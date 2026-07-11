import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./index.css";

// ── Types ─────────────────────────────────────────────────────────────────────

type Role = "user" | "agent" | "error" | "approval" | "system";
type Dict = Record<string, unknown>;

interface Message {
  id:           number;
  role:         Role;
  text:         string;
  timestamp:    string;
  pendingTool?: Dict;
  originalMsg?: string;
  decision?:    "approved" | "rejected";   // stamped on approval cards after a decision
}

interface ApiResponse {
  output:         string;
  pending_tool:   Dict | null;
  needs_approval: boolean;
  error:          string | null;
}

interface HealthResponse {
  status:     string;
  salesforce: string;   // "live" | "mock"
  memory:     string;   // "supabase" | "sqlite"
  qdrant:     string;   // "cloud" | "local"
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getTime(): string {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Session ID persists in localStorage so a page refresh resumes the same
// conversation (the backend remembers it). NEW SESSION mints a fresh one.
function loadSessionId(): string {
  const saved = localStorage.getItem("nexus360_session");
  if (saved) return saved;
  const id = crypto.randomUUID();
  localStorage.setItem("nexus360_session", id);
  return id;
}

// Dev: Vite on :5173 talks to FastAPI on :8000.
// Production: FastAPI serves the built UI itself, so same-origin ("").
const BASE_URL = import.meta.env.DEV ? "http://localhost:8000" : "";

const SUGGESTIONS = [
  "What is the account health for Acme Corp?",
  "List open cases for Globex Inc",
  "Update the Acme Corp - Enterprise License to Closed Won",
  "Create a high priority case for Initech Ltd about API downtime",
];

// ══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════════════════════════

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [health, setHealth]     = useState<HealthResponse | null>(null);

  const [sessionId, setSessionId] = useState<string>(loadSessionId);
  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);
  const idCounter  = useRef(0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Autosize the input as the user types (up to the CSS max-height)
  useEffect(() => {
    const el = inputRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
    }
  }, [input]);

  // Poll backend health so the status dot is honest, not decorative
  useEffect(() => {
    let alive = true;
    async function check() {
      try {
        const { data } = await axios.get<HealthResponse>(`${BASE_URL}/health`);
        if (alive) setHealth(data);
      } catch {
        if (alive) setHealth(null);
      }
    }
    check();
    const timer = setInterval(check, 30_000);
    return () => { alive = false; clearInterval(timer); };
  }, []);

  // ── Add a message ───────────────────────────────────────────────────────────
  function addMessage(role: Role, text: string, extra?: Partial<Message>) {
    setMessages(prev => [...prev, {
      id:        ++idCounter.current,
      role,
      text,
      timestamp: getTime(),
      ...extra,
    }]);
  }

  // ── Handle API response ─────────────────────────────────────────────────────
  function handleApiResponse(data: ApiResponse, originalMsg: string) {
    if (data.error) {
      addMessage("error", `Agent error: ${data.error}`);
      return;
    }

    if (data.needs_approval && data.pending_tool) {
      const tool = data.pending_tool;
      const args = tool.args as Record<string, string>;
      const summary = Object.entries(args)
        .map(([k, v]) => `${k}: ${v}`)
        .join("\n");

      addMessage("approval",
        `⚠️ Write operation requires your approval\n\nTool: ${tool.name}\n${summary}`,
        { pendingTool: tool, originalMsg }
      );
    } else {
      addMessage("agent", data.output);
    }
  }

  // ── Send message ────────────────────────────────────────────────────────────
  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    addMessage("user", trimmed);
    setInput("");
    setLoading(true);

    try {
      const { data } = await axios.post<ApiResponse>(`${BASE_URL}/chat`, {
        message:    trimmed,
        session_id: sessionId,
      });
      handleApiResponse(data, trimmed);
    } catch (err: unknown) {
      const msg = axios.isAxiosError(err) && err.code === "ERR_NETWORK"
        ? "Cannot reach backend. Is the FastAPI server running on port 8000?"
        : (err as Error).message;
      addMessage("error", msg);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  // ── Approve ─────────────────────────────────────────────────────────────────
  // Stamps the card and executes EXACTLY the pending tool via /approve —
  // the card stays in the transcript as an audit trail.
  async function handleApprove(msg: Message) {
    setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, decision: "approved" } : m));
    setLoading(true);

    try {
      const { data } = await axios.post<ApiResponse>(`${BASE_URL}/approve`, {
        message:      msg.originalMsg,
        session_id:   sessionId,
        pending_tool: msg.pendingTool,
      });
      handleApiResponse(data, msg.originalMsg ?? "");
    } catch (err: unknown) {
      addMessage("error", (err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  // ── Reject ──────────────────────────────────────────────────────────────────
  async function handleReject(msg: Message) {
    setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, decision: "rejected" } : m));

    try {
      const { data } = await axios.post<ApiResponse>(`${BASE_URL}/reject`, {
        message:      msg.originalMsg,
        session_id:   sessionId,
        pending_tool: msg.pendingTool,
      });
      addMessage("system", data.output);
    } catch {
      // Backend unreachable — the write was never executed either way
      addMessage("system", "❌ Rejected — action cancelled.");
    }
  }

  // ── Enter key ───────────────────────────────────────────────────────────────
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  // ── New conversation ────────────────────────────────────────────────────────
  function newConversation() {
    const id = crypto.randomUUID();
    localStorage.setItem("nexus360_session", id);
    setSessionId(id);
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  const online = health !== null;

  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <span className="logo-dot" />
          <span className="logo-text">NEXUS<span className="logo-accent">360</span></span>
          <span className="logo-tag">Salesforce AI Agent</span>
          {health?.salesforce === "mock" && (
            <span className="logo-tag logo-mock" title="Salesforce unavailable — running on built-in mock data">
              SF: MOCK
            </span>
          )}
        </div>
        <div className="header-right">
          <span className="session-label">
            SESSION <span className="session-id">{sessionId.slice(0, 8)}...</span>
          </span>
          <button className="new-session-btn" onClick={newConversation}>
            NEW SESSION
          </button>
          <span
            className={`status-dot ${online ? "" : "status-dot--off"}`}
            title={online
              ? `Salesforce: ${health.salesforce} · Memory: ${health.memory} · Qdrant: ${health.qdrant}`
              : "Backend unreachable"}
          />
          <span className={`status-text ${online ? "" : "status-text--off"}`}>
            {online ? "LIVE" : "OFFLINE"}
          </span>
        </div>
      </header>

      {/* ── Chat window ── */}
      <main className="chat-window" aria-live="polite">

        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <p className="empty-title">What do you need to know?</p>
            <p className="empty-sub">Query Salesforce or search internal knowledge base.</p>
            <div className="suggestions">
              {SUGGESTIONS.map(s => (
                <button key={s} className="suggestion-chip" onClick={() => sendMessage(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => {

          if (msg.role === "approval") {
            return (
              <div key={msg.id} className="message message--approval">
                <div className="message-meta">
                  <span className="message-role">APPROVAL REQUIRED</span>
                  <span className="message-time">{msg.timestamp}</span>
                </div>
                <pre className="message-text">{msg.text}</pre>
                {msg.decision ? (
                  <div className={`approval-stamp approval-stamp--${msg.decision}`}>
                    {msg.decision === "approved" ? "✓ APPROVED" : "✕ REJECTED"}
                  </div>
                ) : (
                  <div className="approval-buttons">
                    <button
                      className="approval-btn approval-btn--approve"
                      onClick={() => handleApprove(msg)}
                      disabled={loading}
                    >
                      ✓ APPROVE
                    </button>
                    <button
                      className="approval-btn approval-btn--reject"
                      onClick={() => handleReject(msg)}
                      disabled={loading}
                    >
                      ✕ REJECT
                    </button>
                  </div>
                )}
              </div>
            );
          }

          return (
            <div key={msg.id} className={`message message--${msg.role}`}>
              <div className="message-meta">
                <span className="message-role">
                  {msg.role === "user"   ? "YOU"    :
                   msg.role === "error"  ? "ERROR"  :
                   msg.role === "system" ? "SYSTEM" : "NEXUS360"}
                </span>
                <span className="message-time">{msg.timestamp}</span>
              </div>
              <pre className="message-text">{msg.text}</pre>
            </div>
          );
        })}

        {loading && (
          <div className="message message--agent">
            <div className="message-meta">
              <span className="message-role">NEXUS360</span>
            </div>
            <div className="thinking">
              <span /><span /><span />
              <p>Thinking...</p>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </main>

      {/* ── Input bar ── */}
      <footer className="input-bar">
        <textarea
          ref={inputRef}
          className="input-field"
          rows={1}
          aria-label="Message input"
          placeholder="Ask about accounts, cases, opportunities, or internal policies..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          className="send-btn"
          onClick={() => sendMessage(input)}
          disabled={loading || !input.trim()}
        >
          {loading ? "..." : "SEND"}
        </button>
      </footer>

    </div>
  );
}
