import { useState } from 'react';
import type { AuthUser } from '../auth';
import { authLogin, authRegister } from '../api';

type Tab = 'login' | 'register';

interface Props {
  onSuccess: (user: AuthUser) => void;
  onClose: () => void;
}

export default function AuthModal({ onSuccess, onClose }: Props) {
  const [tab, setTab] = useState<Tab>('login');
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const result = tab === 'login'
        ? await authLogin(username, password)
        : await authRegister(username, email, password);
      onSuccess(result.user);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel auth-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="modal-title">
            {tab === 'login' ? 'Sign in' : 'Create account'}
          </div>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab${tab === 'login' ? ' active' : ''}`}
            onClick={() => { setTab('login'); setError(null); }}
          >
            Sign in
          </button>
          <button
            type="button"
            className={`auth-tab${tab === 'register' ? ' active' : ''}`}
            onClick={() => { setTab('register'); setError(null); }}
          >
            Create account
          </button>
        </div>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">
            <span>{tab === 'login' ? 'Username or email' : 'Username'}</span>
            <input
              type="text"
              className="auth-input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder={tab === 'login' ? 'username or email' : 'username'}
              required
              autoFocus
            />
          </label>

          {tab === 'register' && (
            <label className="auth-label">
              <span>Email</span>
              <input
                type="email"
                className="auth-input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
              />
            </label>
          )}

          <label className="auth-label">
            <span>Password</span>
            <input
              type="password"
              className="auth-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              required
              minLength={tab === 'register' ? 6 : undefined}
            />
          </label>

          {error && <div className="auth-error">{error}</div>}

          <button type="submit" className="primary-btn auth-submit" disabled={loading}>
            {loading ? 'Please wait…' : tab === 'login' ? 'Sign in' : 'Create account'}
          </button>
        </form>
      </div>
    </div>
  );
}
