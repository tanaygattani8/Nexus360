import { useState, useRef, useEffect } from "react";
import axios from "axios";
import "./index.css";

// ── Types ─────────────────────────────────────────────────────────────────────

type Role = "user" | "agent" | "error" | "approval" | "system";
type Dict = Record<string, unknown>;
type ToolType = "READ" | "WRITE" | "RAG";

interface Message {
  id:           number;
  role:         Role;
  text:         string;
  timestamp:    string;
  pendingTool?: Dict;
  originalMsg?: string;
  decision?:    "approved" | "rejected";   // stamped once the backend confirms
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

interface StatsResponse {
  total_runs:         number;
  by_path:            Record<string, number>;
  tool_counts:        Record<string, number>;
  llm_calls_skipped:  number;
  skip_rate:          number;
  est_cost_saved:     number;
  approval_rate:      number;
  writes_approved:    number;
  writes_rejected:    number;
  latency_p50_ms:     number;
  latency_p95_ms:     number;
}

const PATH_LABELS: Record<string, string> = {
  direct:           "Direct reply, no tool",
  template:         "Template, LLM skipped",
  llm:              "LLM synthesis",
  approval_pending: "Write, awaiting approval",
  write_approved:   "Write, approved",
  write_rejected:   "Write, rejected",
};

// ── Icons ─────────────────────────────────────────────────────────────────────
// Inline SVG (Lucide-style geometry). No emoji as icons, no icon-font dependency.

const ICONS: Record<string, string> = {
  check:   "M20 6 9 17l-5-5",
  x:       "M18 6 6 18M6 6l12 12",
  alert:   "M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z",
  send:    "M5 12h14M12 5l7 7-7 7",
  refresh: "M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6",
  plus:    "M12 5v14M5 12h14",
  read:    "M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7-10-7-10-7ZM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6Z",
  write:   "m12 20 9-9-4-4-9 9v4h4ZM15 5l4 4",
  rag:     "M11 3a8 8 0 1 0 0 16 8 8 0 0 0 0-16ZM21 21l-4.3-4.3",
  pulse:   "M22 12h-4l-3 9L9 3l-3 9H2",
  arrow:   "m9 18 6-6-6-6",
  layers:  "m12 2 9 5-9 5-9-5 9-5ZM3 17l9 5 9-5M3 12l9 5 9-5",
};

function Icon({ name, className }: { name: string; className?: string }) {
  return (
    <svg className={`icon${className ? ` ${className}` : ""}`} viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="1.75"
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
      <path d={ICONS[name]} />
    </svg>
  );
}

const TYPE_ICON: Record<ToolType, string> = { READ: "read", WRITE: "write", RAG: "rag" };

// ── Helpers ───────────────────────────────────────────────────────────────────

function getTime(): string {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// Session ID persists in localStorage so a refresh resumes the same conversation.
function loadSessionId(): string {
  const saved = localStorage.getItem("nexus360_session");
  if (saved) return saved;
  const id = crypto.randomUUID();
  localStorage.setItem("nexus360_session", id);
  return id;
}

// Dev: Vite :5173 talks to FastAPI :8000. Production: FastAPI serves the built
// UI itself, so same-origin (""). VITE_API_URL overrides both.
const BASE_URL: string =
  import.meta.env.VITE_API_URL ?? (import.meta.env.DEV ? "http://localhost:8000" : "");

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

// ── Metrics primitives ────────────────────────────────────────────────────────

function Stat({ label, value, sub, tone }: {
  label: string; value: string | number; sub?: string; tone?: "ok" | "warn";
}) {
  return (
    <div className="tile tile--stat">
      <span className="tile-label">{label}</span>
      <span className={`stat-value${tone ? ` stat-value--${tone}` : ""}`}>{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

function Bar({ label, value, max, tone }: {
  label: string; value: number; max: number; tone?: "ok" | "warn";
}) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0;
  return (
    <div className="bar-row">
      <span className="bar-label" title={label}>{label}</span>
      <span className="bar-track">
        <span className={`bar-fill${tone ? ` bar-fill--${tone}` : ""}`} style={{ width: `${pct}%` }} />
      </span>
      <span className="bar-value">{value}</span>
    </div>
  );
}

function MetricsView({ stats, error, onRefresh }: {
  stats: StatsResponse | null; error: boolean; onRefresh: () => void;
}) {
  let body;

  if (error) {
    body = <p className="placeholder">Cannot reach the backend for analytics.</p>;
  } else if (!stats) {
    body = <p className="placeholder">Loading metrics...</p>;
  } else if (stats.total_runs === 0) {
    body = (
      <p className="placeholder">
        No agent runs recorded yet. Ask the agent something on the Console tab, then come back.
      </p>
    );
  } else {
    const pathMax = Math.max(...Object.values(stats.by_path), 1);
    const toolMax = Math.max(...Object.values(stats.tool_counts), 1);
    const paths   = Object.entries(stats.by_path).filter(([, v]) => v > 0);
    const tools   = Object.entries(stats.tool_counts).sort((a, b) => b[1] - a[1]);

    body = (
      <div className="bento bento--metrics">
        <Stat label="TOTAL RUNS" value={stats.total_runs} />
        <Stat label="LLM CALLS SKIPPED" value={stats.llm_calls_skipped}
              sub={`${stats.skip_rate}% of tool turns`} tone="ok" />
        <Stat label="EST. COST AVOIDED" value={`$${stats.est_cost_saved.toFixed(2)}`}
              sub="vs synthesizing every turn" tone="ok" />
        <Stat label="APPROVAL RATE" value={`${stats.approval_rate}%`}
              sub={`${stats.writes_approved} approved / ${stats.writes_rejected} rejected`} />

        <section className="tile tile--wide">
          <span className="tile-label">RESPONSE PATH</span>
          <div className="bars">
            {paths.map(([k, v]) => (
              <Bar key={k} label={PATH_LABELS[k] ?? k} value={v} max={pathMax}
                   tone={k === "template" || k === "direct" ? "ok" : k === "llm" ? "warn" : undefined} />
            ))}
          </div>
        </section>

        <section className="tile tile--wide">
          <span className="tile-label">TOOL USAGE</span>
          <div className="bars">
            {tools.length
              ? tools.map(([k, v]) => <Bar key={k} label={k} value={v} max={toolMax} />)
              : <p className="placeholder placeholder--sm">No tools called yet.</p>}
          </div>
        </section>

        <section className="tile tile--full latency">
          <span className="tile-label">LATENCY</span>
          <div className="latency-row">
            <div><span className="stat-value">{stats.latency_p50_ms}<i>ms</i></span><span className="stat-sub">p50 median</span></div>
            <div><span className="stat-value">{stats.latency_p95_ms}<i>ms</i></span><span className="stat-sub">p95 tail</span></div>
          </div>
        </section>
      </div>
    );
  }

  return (
    <div className="view" aria-live="polite">
      <div className="view-head">
        <div>
          <h1 className="view-title">Agent analytics</h1>
          <p className="view-sub">
            Tool mix, cost avoided, approval rate and latency, measured on this deployment's own runs.
          </p>
        </div>
        <button className="btn" onClick={onRefresh}><Icon name="refresh" />REFRESH</button>
      </div>
      {body}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN
// ══════════════════════════════════════════════════════════════════════════════

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [health, setHealth]     = useState<HealthResponse | null>(null);

  const [view, setView]         = useState<"chat" | "metrics">("chat");
  const [stats, setStats]       = useState<StatsResponse | null>(null);
  const [statsErr, setStatsErr] = useState(false);

  const [sessionId, setSessionId] = useState<string>(loadSessionId);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLTextAreaElement>(null);
  const idCounter = useRef(0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Autosize the composer as the user types (capped by CSS max-height)
  useEffect(() => {
    const el = inputRef.current;
    if (el) {
      el.style.height = "auto";
      el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
    }
  }, [input]);

  // Poll health so the status readout is honest, not decorative
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

  function addMessage(role: Role, text: string, extra?: Partial<Message>) {
    setMessages(prev => [...prev, {
      id: ++idCounter.current, role, text, timestamp: getTime(), ...extra,
    }]);
  }

  function handleApiResponse(data: ApiResponse, originalMsg: string) {
    if (data.error) {
      addMessage("error", data.error);
      return;
    }
    if (data.needs_approval && data.pending_tool) {
      const tool = data.pending_tool;
      const args = tool.args as Record<string, string>;
      const summary = Object.entries(args).map(([k, v]) => `${k}: ${v}`).join("\n");
      addMessage("approval", `${tool.name}\n${summary}`, { pendingTool: tool, originalMsg });
    } else {
      addMessage("agent", data.output);
    }
  }

  async function sendMessage(text: string) {
    const trimmed = text.trim();
    if (!trimmed || loading) return;

    if (view !== "chat") setView("chat");
    addMessage("user", trimmed);
    setInput("");
    setLoading(true);

    try {
      const { data } = await axios.post<ApiResponse>(`${BASE_URL}/chat`, {
        message: trimmed, session_id: sessionId,
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

  // Executes EXACTLY the write the server stashed. Stamp only on success:
  // a stale card (superseded by a newer write) errors and stays actionable.
  async function handleApprove(msg: Message) {
    setLoading(true);
    try {
      const { data } = await axios.post<ApiResponse>(`${BASE_URL}/approve`, {
        message: msg.originalMsg, session_id: sessionId, pending_tool: msg.pendingTool,
      });
      if (!data.error) {
        setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, decision: "approved" } : m));
      }
      handleApiResponse(data, msg.originalMsg ?? "");
    } catch (err: unknown) {
      addMessage("error", (err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function handleReject(msg: Message) {
    setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, decision: "rejected" } : m));
    try {
      const { data } = await axios.post<ApiResponse>(`${BASE_URL}/reject`, {
        message: msg.originalMsg, session_id: sessionId, pending_tool: msg.pendingTool,
      });
      addMessage("system", data.output);
    } catch {
      addMessage("system", "Rejected. The action was not executed.");
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  function newConversation() {
    const id = crypto.randomUUID();
    localStorage.setItem("nexus360_session", id);
    setSessionId(id);
    setMessages([]);
    setInput("");
    setView("chat");
    inputRef.current?.focus();
  }

  async function openMetrics() {
    setView("metrics");
    try {
      const { data } = await axios.get<StatsResponse>(`${BASE_URL}/analytics`);
      setStats(data);
      setStatsErr(false);
    } catch {
      setStatsErr(true);
    }
  }

  // ── Status ──────────────────────────────────────────────────────────────────
  const online = health !== null;

  function statusFor(value: string | undefined, primary: string): "ok" | "warn" | "down" {
    if (!online || !value) return "down";
    return value === primary ? "ok" : "warn";
  }

  const statusRows: { label: string; value: string; state: "ok" | "warn" | "down" }[] = [
    { label: "BACKEND",    value: online ? "LIVE" : "OFFLINE", state: online ? "ok" : "down" },
    { label: "SALESFORCE", value: (health?.salesforce ?? "—").toUpperCase(), state: statusFor(health?.salesforce, "live") },
    { label: "MEMORY",     value: (health?.memory ?? "—").toUpperCase(),     state: statusFor(health?.memory, "supabase") },
    { label: "VECTOR DB",  value: (health?.qdrant ?? "—").toUpperCase(),     state: statusFor(health?.qdrant, "cloud") },
  ];

  const isHome = messages.length === 0 && !loading;

  return (
    <div className="app">

      {/* ── Command bar ── */}
      <header className="topbar">
        <div className="brand">
          <span className={`brand-mark${online ? " brand-mark--live" : ""}`} aria-hidden="true" />
          <span className="brand-name">NEXUS<b>360</b></span>
        </div>

        <nav className="tabs" aria-label="Views">
          <button className={`tab${view === "chat" ? " tab--on" : ""}`}
                  onClick={() => setView("chat")} aria-current={view === "chat"}>
            Console
          </button>
          <button className={`tab${view === "metrics" ? " tab--on" : ""}`}
                  onClick={openMetrics} aria-current={view === "metrics"}>
            Metrics
          </button>
        </nav>

        <div className="topbar-end">
          <span className={`pill pill--${online ? "ok" : "down"}`}>
            <span className={`pip pip--${online ? "ok" : "down"}`} />
            {online ? "LIVE" : "OFFLINE"}
          </span>
          <span className="session" title="Session ID">{sessionId.slice(0, 8)}</span>
          <button className="btn btn--ghost" onClick={newConversation}>
            <Icon name="plus" />NEW
          </button>
        </div>
      </header>

      {/* ── Stage ── */}
      <main className="stage">
        {view === "metrics" ? (
          <MetricsView stats={stats} error={statsErr} onRefresh={openMetrics} />
        ) : isHome ? (
          /* Bento home: capability, live system state, tool inventory, prompts */
          <div className="view">
            <div className="bento bento--home">

              <section className="tile tile--hero" style={{ ["--i" as string]: 0 }}>
                <span className="tile-label">AGENT CONSOLE</span>
                <h1 className="hero-title">
                  Ask the CRM<br /><em>in plain English.</em>
                </h1>
                <p className="hero-sub">
                  Reads accounts, cases and pipeline. Searches internal policy with RAG.
                  Every write stops for your approval before it runs.
                </p>
                <div className="hero-foot">
                  <Icon name="layers" />
                  <span>5 tools, 1 approval gate</span>
                </div>
              </section>

              <section className="tile tile--tall" style={{ ["--i" as string]: 1 }}>
                <span className="tile-label">SYSTEM STATUS</span>
                <ul className="rows">
                  {statusRows.map(row => (
                    <li key={row.label} className="row">
                      <span className="row-key">
                        <span className={`pip pip--${row.state}`} />{row.label}
                      </span>
                      <span className={`row-val row-val--${row.state}`}>{row.value}</span>
                    </li>
                  ))}
                </ul>
                <p className="tile-note">
                  Every service degrades to a local fallback. The demo survives expired free tiers.
                </p>
              </section>

              <section className="tile tile--tall" style={{ ["--i" as string]: 2 }}>
                <span className="tile-label">AGENT TOOLS</span>
                <ul className="rows">
                  {AGENT_TOOLS.map(tool => (
                    <li key={tool.name} className="row">
                      <span className="row-key mono">{tool.name}</span>
                      <span className={`tag tag--${tool.type.toLowerCase()}`}>
                        <Icon name={TYPE_ICON[tool.type]} />{tool.type}
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="tile-note">Writes pause for human approval.</p>
              </section>

              {SUGGESTIONS.map((s, i) => (
                <button key={s.text} className="tile tile--prompt"
                        style={{ ["--i" as string]: 3 + i }}
                        onClick={() => sendMessage(s.text)}>
                  <span className="prompt-head">
                    <span className="tile-label">{s.tag}</span>
                    <span className={`tag tag--${s.type.toLowerCase()}`}>
                      <Icon name={TYPE_ICON[s.type]} />{s.type}
                    </span>
                  </span>
                  <span className="prompt-text">{s.text}</span>
                  <span className="prompt-go"><Icon name="arrow" /></span>
                </button>
              ))}

            </div>
          </div>
        ) : (
          /* Conversation */
          <div className="thread" aria-live="polite">
            {messages.map(msg => {

              if (msg.role === "approval") {
                const [toolName, ...argLines] = msg.text.split("\n");
                return (
                  <article key={msg.id} className="msg msg--approval">
                    <header className="approval-head">
                      <Icon name="alert" />
                      <span>Approval required</span>
                      <time>{msg.timestamp}</time>
                    </header>
                    <div className="approval-body">
                      <span className="approval-tool mono">{toolName}</span>
                      <dl className="approval-args">
                        {argLines.map(line => {
                          const idx = line.indexOf(":");
                          return (
                            <div key={line} className="arg">
                              <dt>{line.slice(0, idx)}</dt>
                              <dd className="mono">{line.slice(idx + 1).trim()}</dd>
                            </div>
                          );
                        })}
                      </dl>
                    </div>
                    {msg.decision ? (
                      <div className={`stamp stamp--${msg.decision}`}>
                        <Icon name={msg.decision === "approved" ? "check" : "x"} />
                        {msg.decision === "approved" ? "APPROVED, EXECUTED" : "REJECTED, NOT EXECUTED"}
                      </div>
                    ) : (
                      <div className="approval-actions">
                        <button className="btn btn--go" onClick={() => handleApprove(msg)} disabled={loading}>
                          <Icon name="check" />APPROVE
                        </button>
                        <button className="btn btn--no" onClick={() => handleReject(msg)} disabled={loading}>
                          <Icon name="x" />REJECT
                        </button>
                      </div>
                    )}
                  </article>
                );
              }

              return (
                <article key={msg.id} className={`msg msg--${msg.role}`}>
                  <header className="msg-head">
                    <span className="msg-who">
                      {msg.role === "user"   ? "YOU"    :
                       msg.role === "error"  ? "ERROR"  :
                       msg.role === "system" ? "SYSTEM" : "NEXUS360"}
                    </span>
                    <time>{msg.timestamp}</time>
                  </header>
                  <pre className="msg-body">{msg.text}</pre>
                </article>
              );
            })}

            {loading && (
              <article className="msg msg--agent">
                <header className="msg-head"><span className="msg-who">NEXUS360</span></header>
                <div className="thinking"><i /><i /><i /></div>
              </article>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </main>

      {/* ── Composer ── */}
      {view === "chat" && (
        <footer className="composer">
          <div className="composer-inner">
            <textarea
              ref={inputRef}
              className="composer-input"
              rows={1}
              aria-label="Message the agent"
              placeholder="Ask about accounts, cases, opportunities, or internal policy..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button className="btn btn--send" onClick={() => sendMessage(input)}
                    disabled={loading || !input.trim()} aria-label="Send message">
              <Icon name="send" />
            </button>
          </div>
          <p className="composer-hint">
            Enter to send, Shift+Enter for a new line. Writes always ask first.
          </p>
        </footer>
      )}
    </div>
  );
}
