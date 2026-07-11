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
  qdrant:     string;   // "cloud" | "local" | "bm25-lite"
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
// VITE_API_URL overrides both (for a statically hosted frontend).
const BASE_URL: string =
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

type ToolType = "READ" | "WRITE" | "RAG";

const SUGGESTIONS: { tag: string; type: ToolType; text: string }[] = [
  { tag: "ACCOUNT HEALTH", type: "READ",  text: "What is the account health for Acme Corp?" },
  { tag: "KNOWLEDGE BASE", type: "RAG",   text: "What is the escalation policy for high priority cases?" },
  { tag: "PIPELINE",       type: "WRITE", text: "Update the Acme Corp - Enterprise License to Closed Won" },
  { tag: "SUPPORT",        type: "WRITE", text: "Create a high priority case for Initech Ltd about API downtime" },
];

const AGENT_TOOLS: { name: string; type: ToolType }[] = [
  { name: "get_account_health",       type: "READ" },
  { name: "list_open_cases",          type: "READ" },
  { name: "search_knowledge_base",    type: "RAG" },
  { name: "update_opportunity_stage", type: "WRITE" },
  { name: "create_support_case",      type: "WRITE" },
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

  // Poll backend health so the status panel is honest, not decorative
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

  // Status row: green when the primary service is live, amber on a local
  // fallback, red when the backend itself is unreachable.
  function statusFor(value: string | undefined, primary: string): "ok" | "fallback" | "down" {
    if (!online || !value) return "down";
    return value === primary ? "ok" : "fallback";
  }

  const statusRows: { label: string; value: string; state: "ok" | "fallback" | "down" }[] = [
    { label: "BACKEND",    value: online ? "LIVE" : "OFFLINE", state: online ? "ok" : "down" },
    { label: "SALESFORCE", value: (health?.salesforce ?? "—").toUpperCase(), state: statusFor(health?.salesforce, "live") },
    { label: "MEMORY",     value: (health?.memory ?? "—").toUpperCase(),     state: statusFor(health?.memory, "supabase") },
    { label: "VECTOR DB",  value: (health?.qdrant ?? "—").toUpperCase(),     state: statusFor(health?.qdrant, "cloud") },
  ];

  return (
    <div className="app">

      {/* ── Sidebar (desktop) ── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <span className="logo-dot" />
          <div>
            <div className="logo-text">NEXUS<span className="logo-accent">360</span></div>
            <div className="logo-sub">SALESFORCE AI AGENT</div>
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">SYSTEM STATUS</div>
          {statusRows.map(row => (
            <div key={row.label} className="status-row">
              <span className="status-row-label">
                <span className={`status-pip status-pip--${row.state}`} />
                {row.label}
              </span>
              <span className={`status-row-value status-row-value--${row.state}`}>{row.value}</span>
            </div>
          ))}
        </div>

        <div className="panel">
          <div className="panel-title">AGENT TOOLS</div>
          {AGENT_TOOLS.map(tool => (
            <div key={tool.name} className="tool-row">
              <span className="tool-name">{tool.name}</span>
              <span className={`tag tag--${tool.type.toLowerCase()}`}>{tool.type}</span>
            </div>
          ))}
          <div className="panel-note">Writes pause for human approval before executing.</div>
        </div>

        <div className="sidebar-foot">
          <div className="session-label">
            SESSION <span className="session-id">{sessionId.slice(0, 8)}</span>
          </div>
          <button className="new-session-btn" onClick={newConversation}>
            NEW SESSION
          </button>
        </div>
      </aside>

      {/* ── Workspace ── */}
      <div className="workspace">

        {/* Compact topbar — replaces the sidebar on small screens */}
        <header className="topbar">
          <div className="sidebar-brand">
            <span className="logo-dot" />
            <span className="logo-text">NEXUS<span className="logo-accent">360</span></span>
          </div>
          <div className="topbar-right">
            <span
              className={`status-pip status-pip--${online ? "ok" : "down"}`}
              title={online ? "Backend online" : "Backend unreachable"}
            />
            <span className={`status-row-value status-row-value--${online ? "ok" : "down"}`}>
              {online ? "LIVE" : "OFFLINE"}
            </span>
            <button className="new-session-btn" onClick={newConversation}>NEW SESSION</button>
          </div>
        </header>

        {/* ── Chat window ── */}
        <main className="chat-window" aria-live="polite">

          {messages.length === 0 && !loading && (
            <div className="empty-state">
              <p className="empty-eyebrow">AGENT CONSOLE</p>
              <h1 className="empty-title">What do you need to know?</h1>
              <p className="empty-sub">
                Query live Salesforce data, search internal policies, or update the pipeline —
                write operations always ask for your approval first.
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map(s => (
                  <button key={s.text} className="suggestion-card" onClick={() => sendMessage(s.text)}>
                    <span className="suggestion-tags">
                      <span className="tag">{s.tag}</span>
                      <span className={`tag tag--${s.type.toLowerCase()}`}>{s.type}</span>
                    </span>
                    <span className="suggestion-text">{s.text}</span>
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
          <div className="input-inner">
            <span className="prompt-glyph" aria-hidden="true">❯</span>
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
          </div>
        </footer>

      </div>
    </div>
  );
}
