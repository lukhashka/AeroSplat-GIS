import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState, type FormEvent } from "react";

import { auth, UNAUTHORIZED_EVENT } from "../api/client";
import { ErrorBox, Modal } from "./Modal";

/**
 * Asks for an access token when the API answers 401 — i.e. the page is opened from another
 * machine (aerosplat.server.auth). On this machine it never shows: loopback needs no token.
 * The token goes to POST /auth/login once and comes back as an HttpOnly cookie; the page
 * itself does not keep it.
 */
export function AuthGate() {
  const qc = useQueryClient();
  const [reason, setReason] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const on = (e: Event) => setReason((prev) => prev ?? String((e as CustomEvent).detail ?? ""));
    window.addEventListener(UNAUTHORIZED_EVENT, on);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, on);
  }, []);

  if (reason === null) return null;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!token.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      await auth.login(token.trim());
      setToken("");
      setReason(null);
      await qc.invalidateQueries();
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      title="Потрібен токен доступу"
      subtitle="Сервер приймає запити без токена лише з власної машини."
      onClose={() => undefined}
      footer={
        <button className="btn primary" type="submit" form="auth-form" disabled={!token.trim() || busy}>
          Увійти
        </button>
      }
    >
      <form id="auth-form" onSubmit={submit}>
        <label className="field full">
          <span>Токен</span>
          <input
            type="password"
            autoComplete="off"
            autoFocus
            value={token}
            placeholder="asg_…"
            onChange={(e) => setToken(e.target.value)}
            aria-label="Токен доступу"
          />
        </label>
      </form>
      <div className="note">
        Токен видає власник сервера командою <code>aerosplat token create &lt;ім'я&gt;</code>. Ваші зміни в
        журналі аудиту підписуються іменем токена.
      </div>
      {reason && <div className="faint" style={{ fontSize: 12 }}>{reason}</div>}
      <ErrorBox error={error} />
    </Modal>
  );
}
