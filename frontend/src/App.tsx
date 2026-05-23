import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./index.css";

// ── Types ─────────────────────────────────────────────────────────────────────

type Role = "user" | "agent" | "error" | "approval" | "system";

interface Message {
  id:           number;
  role:         Role;
  text:         string;
  timestamp:    string;
  pendingTool?: dict;
  originalMsg?: string;
}

interface ApiResponse {
  output:         string;
  pending_tool:   dict | null;
  needs_approval: boolean;
  error:          string | null;
}

type dict = Record<string, unknown>;

// ── Helpers ───────────────────────────────────────────────────────────────────

function getTime(): string {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Generate a UUID for this browser session.
// crypto.randomUUID() is built into all modern browsers — no library needed.
// Called once when the app loads. Every message in this tab uses the same UUID.
// New tab = new UUID = new conversation = fresh memory.
function generateSessionId(): string {
  return crypto.randomUUID();
}

const BASE_URL = "http://localhost:8000";

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

  // session_id is generated once when the component mounts.
  // useRef keeps it stable across re-renders — useState would also work but
  // useRef makes it clear this value never changes during the session.
  const sessionId  = useRef<string>(generateSessionId());
  const bottomRef  = useRef<HTMLDivElement>(null);
  const inputRef   = useRef<HTMLTextAreaElement>(null);
  const idCounter  = useRef(0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

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
      const tool = data.pending_tool as dict;
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
        session_id: sessionId.current,   // ← send session UUID with every message
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
  async function handleApprove(msg: Message) {
    setMessages(prev => prev.filter(m => m.id !== msg.id));
    addMessage("system", `✅ Approved — executing ${(msg.pendingTool as dict).name as string}...`);
    setLoading(true);

    try {
      const { data } = await axios.post<ApiResponse>(`${BASE_URL}/approve`, {
        message:      msg.originalMsg,
        session_id:   sessionId.current,   // ← same session UUID
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
    setMessages(prev => prev.filter(m => m.id !== msg.id));
    addMessage("system", "❌ Rejected — action cancelled.");
  }

  // ── Enter key ───────────────────────────────────────────────────────────────
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  // ── New conversation ────────────────────────────────────────────────────────
  // Generates a new UUID and clears the chat — starts a fresh memory session.
  function newConversation() {
    sessionId.current = generateSessionId();
    setMessages([]);
    setInput("");
    inputRef.current?.focus();
  }

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <div className="app">

      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <span className="logo-dot" />
          <span className="logo-text">NEXUS<span className="logo-accent">360</span></span>
          <span className="logo-tag">Salesforce AI Agent</span>
          <span className="logo-tag logo-phase">PHASE 3</span>
        </div>
        <div className="header-right">
          {/* Session ID shown in header — useful for demos to prove memory is per-session */}
          <span className="session-label">
            SESSION <span className="session-id">{sessionId.current.slice(0, 8)}...</span>
          </span>
          <button className="new-session-btn" onClick={newConversation}>
            NEW SESSION
          </button>
          <span className="status-dot" />
          <span className="status-text">LIVE</span>
        </div>
      </header>

      {/* ── Chat window ── */}
      <main className="chat-window">

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
              <p>Querying Salesforce...</p>
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
          placeholder="Ask about accounts, cases, opportunities, or internal policies..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
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
